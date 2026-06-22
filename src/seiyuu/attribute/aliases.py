"""Deterministic character alias resolution — a once-per-book post-pass over the FULL
registry (it needs every chapter integrated to judge whether a surname is contested).

Small models leave honorific variants un-merged (`Darcy` vs `Mr. Darcy`). This pass merges
only what is provably the same person and FLAGS everything ambiguous, honoring SPEC's
precision-over-recall rule (over-merging two distinct characters is the worst failure):

- **Auto-merge** (the only two): honorific-strip exact match, and subsumed-alias
  consolidation (one record's whole name set is a subset of another's). Both gated by a
  gender/generation guard.
- **Flag-only** (never merged): honorific groups with a gender/generation conflict
  (`Mr.`/`Mrs. Bennet`, `Lady`/`Miss Lucas`), and low-evidence records (no metadata, no
  attributed lines — possible hallucinations). First names and nicknames are never
  auto-merged: a bare given name simply doesn't match a full name here, so it stays put.

`resolve_chunk` (the incremental, per-chunk, cache-reproducible path) is untouched. The
``resolver`` seam is reserved for a future opt-in LLM adjudication pass; it is not built or
called here.
"""

from collections import Counter, defaultdict
from typing import Protocol

from seiyuu.attribute.models import AttributedChapter, Character, CharacterRegistry

_TITLES = frozenset(
    {
        "mr", "mr.", "mister", "mrs", "mrs.", "miss", "miss.", "ms", "ms.",
        "master", "lady", "lord", "sir", "dr", "dr.", "doctor", "madam", "mistress",
    }
)  # fmt: skip

# Title -> (gender/role) class for the conflict guard. None of these classes are
# compatible with a *different* non-"none" class, so Mr./Mrs. and Lady/Miss never merge.
_MALE = {"mr", "mr.", "mister", "sir", "lord"}
_FEMALE_ADULT = {"mrs", "mrs.", "lady", "madam", "mistress"}
_FEMALE_UNMARRIED = {"miss", "miss.", "ms", "ms."}
_CHILD = {"master", "young"}


class AliasResolver(Protocol):
    """Seam for a future opt-in LLM adjudication of flagged candidates (not built yet)."""

    def resolve(
        self, candidates: list[str], registry: CharacterRegistry
    ) -> list[tuple[str, str]]: ...


def _strip_title(name: str) -> str:
    toks = name.casefold().split()
    if toks and toks[0] in _TITLES:
        toks = toks[1:]
    return " ".join(toks)


def _title_class(name: str) -> str:
    tok0 = (name.casefold().split() or [""])[0]
    if tok0 in _MALE:
        return "male_adult"
    if tok0 in _FEMALE_ADULT:
        return "female_adult"
    if tok0 in _FEMALE_UNMARRIED:
        return "female_unmarried"
    if tok0 in _CHILD:
        return "child"
    return "none"


def _norm_gender(g: str | None) -> str:
    if not g:
        return "unknown"
    g = g.strip().casefold()
    if g in {"male", "m", "man", "boy"}:
        return "male"
    if g in {"female", "f", "woman", "girl"}:
        return "female"
    return "unknown"


def _conflict(a: Character, b: Character) -> bool:
    """True if a and b cannot be the same person (gender or generation/role clash)."""
    ga, gb = _norm_gender(a.gender), _norm_gender(b.gender)
    if ga != "unknown" and gb != "unknown" and ga != gb:
        return True
    ca, cb = _title_class(a.canonical_name), _title_class(b.canonical_name)
    return ca != "none" and cb != "none" and ca != cb


def _names(c: Character) -> set[str]:
    return {n.casefold() for n in (c.canonical_name, *c.aliases)}


def _winner(members: list[Character]) -> Character:
    # Most declared aliases, then earliest first_appearance, then smallest id — deterministic
    # so attribution.json does not churn across reruns or input order.
    return min(
        members,
        key=lambda c: (-len(c.aliases), c.first_appearance or "~", c.id),
    )


def _flatten(remap: dict[str, str]) -> dict[str, str]:
    """Collapse transitive chains (A->B, B->C => A->C) to final survivors."""

    def root(x: str) -> str:
        seen: set[str] = set()
        while x in remap and x not in seen:
            seen.add(x)
            x = remap[x]
        return x

    return {loser: root(loser) for loser in remap}


def resolve_registry_aliases(
    registry: CharacterRegistry,
    chapters: list[AttributedChapter],
    *,
    resolver: AliasResolver | None = None,
) -> tuple[dict[str, str], list[str]]:
    """Merge provably-same characters, flag the ambiguous. Mutates ``registry`` in place.

    Returns ``(id_remap, notes)``: ``id_remap`` maps each absorbed character id to its
    surviving id (the caller rewrites segment speakers); ``notes`` are review flags.
    """
    seg_count: Counter[str] = Counter()
    for chapter in chapters:
        for seg in chapter.segments:
            if seg.speaker:
                seg_count[seg.speaker] += 1

    remap: dict[str, str] = {}
    notes: list[str] = []

    # Rule 1 — honorific-strip exact match: group by the title-stripped canonical name.
    groups: dict[str, list[Character]] = defaultdict(list)
    for char in registry.characters:
        key = _strip_title(char.canonical_name)
        if len(key) >= 3 or " " in key:  # avoid merging on a tiny fragment
            groups[key].append(char)
    for key, members in groups.items():
        if len(members) < 2:
            continue
        if any(_conflict(a, b) for i, a in enumerate(members) for b in members[i + 1 :]):
            others = ", ".join(repr(m.canonical_name) for m in members)
            notes.append(f"alias: ambiguous '{key}' shared by [{others}] — not merged, review")
            continue
        winner = _winner(members)
        for m in members:
            if m is not winner:
                remap[m.id] = winner.id

    # Rule 2 — subsumed-alias consolidation: B's whole name set is a subset of A's. Dedup of
    # an identity the model already asserted, not new inference. Only among not-yet-merged.
    free = [c for c in registry.characters if c.id not in remap]
    for a in free:
        for b in free:
            if a is b or a.id in remap or b.id in remap:
                continue
            if _names(b) <= _names(a) and not _conflict(a, b):
                remap[b.id] = a.id

    remap = _flatten(remap)
    _apply_merges(registry, remap)

    # Flag low-evidence records (no metadata, no attributed lines) — possible hallucinations.
    for char in registry.characters:
        if (
            seg_count.get(char.id, 0) == 0
            and char.gender is None
            and char.age_hint is None
            and char.description is None
        ):
            notes.append(
                f"alias: low-evidence character {char.id!r} ({char.canonical_name!r}) — "
                f"no metadata, no attributed lines, possible hallucination, review"
            )

    # `resolver` (future LLM adjudication of the flagged candidates) is a deferred seam.
    return remap, notes


def _apply_merges(registry: CharacterRegistry, remap: dict[str, str]) -> None:
    by_id = {c.id: c for c in registry.characters}
    folded: dict[str, list[str]] = defaultdict(list)
    for loser, winner in remap.items():
        folded[winner].append(loser)
    for winner_id, losers in folded.items():
        winner = by_id[winner_id]
        for loser_id in losers:
            loser = by_id[loser_id]
            for name in (loser.canonical_name, *loser.aliases):
                if not winner.matches_name(name) and name not in winner.aliases:
                    winner.aliases.append(name)
            winner.gender = winner.gender or loser.gender
            winner.age_hint = winner.age_hint or loser.age_hint
            winner.description = winner.description or loser.description
            if loser.first_appearance and (
                winner.first_appearance is None or loser.first_appearance < winner.first_appearance
            ):
                winner.first_appearance = loser.first_appearance
    registry.characters = [c for c in registry.characters if c.id not in remap]
