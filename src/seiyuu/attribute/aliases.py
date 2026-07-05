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

`resolve_chunk` (the incremental, per-chunk, cache-reproducible path) is untouched.

The ``resolver`` seam is the opt-in LLM adjudication pass (default OFF: ``resolver=None`` is
byte-identical to the deterministic-only behavior). When present, it resolves ONLY the gray
zone the two auto-rules skip — first-name<->full-name, title<->given, curated nicknames —
over a small, deterministically-generated, ``_conflict``-clean candidate set it can only
APPROVE/REJECT. The cardinal failure (over-merging two distinct characters) is guarded at
GENERATION (the sibling-trap and ambiguous-leader cases never become candidates) and again
by a hard ``_conflict`` veto re-applied AFTER the resolver returns, so a buggy or adversarial
resolver can never bypass it.
"""

from collections import Counter, defaultdict
from typing import Protocol

from seiyuu.attribute.models import (
    AttributedChapter,
    CandidatePair,
    Character,
    CharacterEvidence,
    CharacterRegistry,
    PairVerdict,
)
from seiyuu.attribute.nicknames import is_nickname_pair

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
    """Opt-in LLM adjudication seam: approve/reject deterministically-generated candidates.

    The typed contract is the precision lever — the resolver is handed pre-generated,
    ``_conflict``-clean :class:`CandidatePair`s and returns a :class:`PairVerdict` per pair.
    It can NEVER emit a character id/name or introduce a pair the generator did not surface;
    the merge itself (survivor selection, the hard veto) stays in ``resolve_registry_aliases``.
    """

    def resolve(
        self, candidates: list[CandidatePair], registry: CharacterRegistry
    ) -> list[PairVerdict]: ...


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


def _tokens(name: str) -> list[str]:
    return name.casefold().split()


def _is_bare_given(c: Character) -> bool:
    """True if the record's entire name-set is a single, title-free token (e.g. `Elizabeth`)."""
    names = _names(c)
    if len(names) != 1:
        return False
    tok = next(iter(names))
    return " " not in tok and tok not in _TITLES


def _given_token(c: Character) -> str | None:
    """The leading given-name token of the canonical name (None if it leads with a title)."""
    toks = _tokens(c.canonical_name)
    if toks and toks[0] not in _TITLES:
        return toks[0]
    return None


def _title_surname(c: Character) -> str | None:
    """Surname if the canonical name is exactly Title + Surname with no given name."""
    toks = _tokens(c.canonical_name)
    if len(toks) == 2 and toks[0] in _TITLES:
        return toks[1]
    return None


def _given_surname(c: Character) -> str | None:
    """Surname if the canonical name is Given (+ ...) + Surname led by a non-title token."""
    toks = _tokens(c.canonical_name)
    if len(toks) >= 2 and toks[0] not in _TITLES:
        return toks[-1]
    return None


def _surname_or_none(c: Character) -> str | None:
    """Trailing surname of a given-led multi-token name (used to guard nickname pairs)."""
    return _given_surname(c)


def _pair_id(a: Character, b: Character) -> str:
    lo, hi = sorted((a.id, b.id))
    return f"{lo}::{hi}"


def _make_pair(a: Character, b: Character, generator: str) -> CandidatePair:
    # Canonical a/b order (by id) so pair_id and the candidates_digest are rerun-stable.
    lo, hi = (a, b) if a.id <= b.id else (b, a)
    return CandidatePair(
        pair_id=_pair_id(a, b),
        generator=generator,
        a=CharacterEvidence.from_character(lo),
        b=CharacterEvidence.from_character(hi),
    )


def _generate_candidates(
    registry: CharacterRegistry,
    seg_count: Counter[str],
    *,
    cap: int,
    use_nicknames: bool,
) -> tuple[list[CandidatePair], list[str]]:
    """Deterministically surface the gray-zone merge candidates the auto-rules skip.

    Runs AFTER Rule 1/Rule 2 merged and ``_apply_merges`` mutated ``registry`` (so no already
    -merged record is proposed). Three generators (G1 given-name containment, G2 restricted
    title+surname, G3 curated nicknames) each emit ``_conflict``-clean pairs; ambiguous cases
    (a bare given name leading 2+ full names; a title matching 2+ given names) are FLAGGED,
    never turned into a merge candidate — the sibling-trap is closed here, not delegated to
    the LLM. Returns ``(candidates, flag_notes)``; ``candidates`` is deduped, ``_conflict``-
    clean, ordered G1<G2<G3, and capped at ``cap`` (overflow flagged, not paid for).
    """
    chars = registry.characters
    flags: list[str] = []
    ordered: list[CandidatePair] = []
    seen: set[frozenset[str]] = set()

    def emit(a: Character, b: Character, generator: str) -> None:
        key = frozenset({a.id, b.id})
        if a.id == b.id or key in seen:
            return
        if _conflict(a, b):  # honorific/gender clash — those stay flag-only, never adjudicated
            return
        seen.add(key)
        ordered.append(_make_pair(a, b, generator))

    # G1 — bare given name <-> full name it uniquely leads.
    for c in chars:
        if not _is_bare_given(c):
            continue
        token = next(iter(_names(c)))
        leaders = [
            other
            for other in chars
            if other.id != c.id
            and any(len(toks) >= 2 and toks[0] == token for toks in map(_tokens, _names(other)))
        ]
        if len(leaders) == 1:
            emit(c, leaders[0], "G1")
        elif len(leaders) >= 2:
            names = ", ".join(repr(x.canonical_name) for x in leaders)
            flags.append(
                f"alias: ambiguous given name '{token}' leads [{names}] — not merged, review"
            )

    # G2 — Title+Surname <-> Given+Surname (RESTRICTED: never two given-name records that
    # share a surname — that sibling case is exactly what _conflict cannot catch).
    given_by_surname: dict[str, list[Character]] = defaultdict(list)
    for c in chars:
        surname = _given_surname(c)
        if surname is not None:
            given_by_surname[surname].append(c)
    for c in chars:
        surname = _title_surname(c)
        if surname is None:
            continue
        matches = given_by_surname.get(surname, [])
        if len(matches) == 1:
            emit(c, matches[0], "G2")
        elif len(matches) >= 2:
            names = ", ".join(repr(x.canonical_name) for x in matches)
            flags.append(
                f"alias: ambiguous title '{c.canonical_name}' matches [{names}] "
                f"— not merged, review"
            )

    # G3 — curated nickname/diminutive table (fuzzy/edit-distance stays OFF).
    if use_nicknames:
        for i, a in enumerate(chars):
            ga = _given_token(a)
            if ga is None:
                continue
            for b in chars[i + 1 :]:
                gb = _given_token(b)
                if gb is None or not is_nickname_pair(ga, gb):
                    continue
                sa, sb = _surname_or_none(a), _surname_or_none(b)
                if sa is not None and sb is not None and sa != sb:
                    continue  # distinct surnames -> not the same person, don't even propose
                emit(a, b, "G3")

    # Priority + evidence ordering for the cap: G1 first, then more-attested pairs.
    rank = {"G1": 0, "G2": 1, "G3": 2}

    def evidence(pair: CandidatePair) -> int:
        return seg_count.get(pair.a.id, 0) + seg_count.get(pair.b.id, 0)

    ordered.sort(key=lambda p: (rank[p.generator], -evidence(p), p.pair_id))
    if len(ordered) > cap:
        dropped = ordered[cap:]
        ordered = ordered[:cap]
        flags.append(
            f"alias: {len(dropped)} adjudication candidate(s) over the cap of {cap} "
            f"were not sent for review this run"
        )
    return ordered, flags


def resolve_registry_aliases(
    registry: CharacterRegistry,
    chapters: list[AttributedChapter],
    *,
    resolver: AliasResolver | None = None,
    confidence_threshold: float = 0.85,
    candidate_cap: int = 40,
    use_nicknames: bool = True,
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

    # Opt-in LLM adjudication of the gray zone the auto-rules skip (default OFF: resolver=None
    # leaves everything below untouched, so the output is byte-identical to today).
    if resolver is not None:
        adj_remap, adj_notes = _adjudicate(
            registry,
            seg_count,
            resolver=resolver,
            confidence_threshold=confidence_threshold,
            candidate_cap=candidate_cap,
            use_nicknames=use_nicknames,
        )
        notes.extend(adj_notes)
        if adj_remap:
            remap = _flatten({**remap, **adj_remap})
            _apply_merges(registry, adj_remap)

    # Flag low-evidence records (no metadata, no attributed lines) — possible hallucinations.
    # Computed on the FINAL registry so an adjudication-merged record leaves no stale flag.
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

    return remap, notes


def _adjudicate(
    registry: CharacterRegistry,
    seg_count: Counter[str],
    *,
    resolver: AliasResolver,
    confidence_threshold: float,
    candidate_cap: int,
    use_nicknames: bool,
) -> tuple[dict[str, str], list[str]]:
    """Generate candidates, ask the resolver, and turn approvals into a merge remap.

    Precision is layered: only ``_conflict``-clean generated pairs are shown; a verdict must
    be ``same_person`` AND clear the confidence threshold; ``_conflict`` is re-applied as a
    HARD VETO here (never inside the resolver), so a buggy/adversarial resolver cannot bypass
    it. Approved pairs pick a deterministic survivor via ``_winner`` and feed the same remap
    machinery as the auto-rules, keeping ``attribution.json`` stable across reruns.
    """
    candidates, notes = _generate_candidates(
        registry, seg_count, cap=candidate_cap, use_nicknames=use_nicknames
    )
    if not candidates:
        return {}, notes

    verdicts = resolver.resolve(candidates, registry)
    verdict_by_id: dict[str, PairVerdict] = {v.pair_id: v for v in verdicts}
    by_id = {c.id: c for c in registry.characters}

    adj_remap: dict[str, str] = {}
    for cand in candidates:
        a, b = by_id.get(cand.a.id), by_id.get(cand.b.id)
        if a is None or b is None or a.id in adj_remap or b.id in adj_remap:
            continue  # a member was already absorbed by an earlier approval this pass
        pair_desc = f"{a.canonical_name!r} <-> {b.canonical_name!r}"
        verdict = verdict_by_id.get(cand.pair_id)
        if verdict is None or not verdict.same_person:
            notes.append(f"alias: adjudicator rejected {pair_desc} ({cand.generator}) — not merged")
            continue
        if verdict.confidence < confidence_threshold:
            notes.append(
                f"alias: adjudicator approved {pair_desc} at confidence {verdict.confidence:.2f} "
                f"< {confidence_threshold:.2f} — flagged, not merged"
            )
            continue
        if _conflict(a, b):  # HARD VETO, re-applied after the resolver — never bypassable
            notes.append(f"alias: vetoed {pair_desc} despite approval — gender/generation clash")
            continue
        winner = _winner([a, b])
        loser = b if winner is a else a
        adj_remap[loser.id] = winner.id
        notes.append(
            f"alias: merged {loser.canonical_name!r} -> {winner.canonical_name!r} "
            f"(LLM {cand.generator}, conf {verdict.confidence:.2f})"
        )
    return adj_remap, notes


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
