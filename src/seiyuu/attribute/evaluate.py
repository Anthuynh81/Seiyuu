"""Attribution ACCURACY scoring: gold quote labels vs an attribution report (pure, no I/O
beyond the explicit JSONL helpers).

The unit of evaluation is one QUOTED span — exactly what the pipeline's deterministic
splitter (``attribute/spans.py``) hands the model and what one speaker label decides. A gold
row identifies its quote by ``block_id`` + ``quote_index`` (the 0-based ordinal among the
block's quoted spans, the shared :func:`~seiyuu.attribute.spans.quoted_ordinals` definition)
and carries the quote ``text`` plus char offsets so a re-ingested book can still be realigned.

Speaker matching goes through the ARM's registry, never raw ids: a predicted character maps
to the gold speaker whose name set (canonical + aliases) it matches — exactly first
(case/punctuation-insensitive), then honorific-stripped with a title-class guard so
``Mr. Bennet`` and ``Mrs. Bennet`` never collapse onto each other. A predicted character that
matches two gold speakers equally well is AMBIGUOUS and scores wrong (precision over recall,
same as ``aliases.py``). An unattributed quote (narration fallback) is always wrong.

``PDNC``-style quote types (explicit / anaphoric / implicit) are optional per row; the
breakdown reports an ``unlabeled`` bucket for rows without one.
"""

import re
from collections import Counter, defaultdict
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, computed_field, field_validator

from seiyuu.attribute.models import AttributionReport, Segment, SegmentType
from seiyuu.attribute.spans import (
    DialogueConvention,
    detect_dialogue_convention,
    is_quoted_span,
    quoted_ordinals,
    thought_candidate_spans,
)
from seiyuu.ingest.models import BlockType, NormalizedBook

UNATTRIBUTED = "<unattributed>"
UNALIGNED = "<unaligned>"
UNLABELED_TYPE = "unlabeled"


class QuoteType(StrEnum):
    """PDNC's quote taxonomy: speaker named in the attribution tag, referred to by a pronoun
    or description, or left to context (turn-taking)."""

    EXPLICIT = "explicit"
    ANAPHORIC = "anaphoric"
    IMPLICIT = "implicit"


class GoldQuote(BaseModel):
    """One gold-labeled quoted span (one JSONL row)."""

    book_id: str
    chapter_index: int = Field(ge=1)  # 1-based, as in AttributedChapter.index
    block_id: str
    quote_index: int = Field(ge=0)  # ordinal among the block's QUOTED spans
    char_start: int | None = Field(default=None, ge=0)  # offset into the block text
    char_end: int | None = Field(default=None, ge=0)
    text: str  # the quoted span verbatim (realignment + human readability)
    speaker: str  # gold speaker canonical name
    aliases: list[str] = []
    quote_type: QuoteType | None = None

    @field_validator("speaker")
    @classmethod
    def _speaker_nonempty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("gold speaker must be a non-empty name")
        return value

    @field_validator("quote_type", mode="before")
    @classmethod
    def _fold_type(cls, value: object) -> object:
        # PDNC writes "Explicit"/"Anaphoric"/"Implicit"; empty means unlabeled.
        if isinstance(value, str):
            value = value.strip().casefold()
            return value or None
        return value


# --------------------------------------------------------------------------- JSONL I/O


def load_gold(path: Path) -> list[GoldQuote]:
    """Read a gold JSONL file; a bad row fails loudly with its line number."""
    path = Path(path)
    rows: list[GoldQuote] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(GoldQuote.model_validate_json(line))
        except ValidationError as exc:
            raise ValueError(f"{path}:{lineno}: invalid gold row: {exc}") from exc
    keys = Counter((r.block_id, r.quote_index) for r in rows)
    dupes = sorted(k for k, n in keys.items() if n > 1)
    if dupes:
        raise ValueError(f"{path}: duplicate gold rows for (block_id, quote_index) {dupes[:5]}")
    return rows


def dump_gold(rows: list[GoldQuote]) -> str:
    return "".join(r.model_dump_json(exclude_none=True) + "\n" for r in rows)


# ----------------------------------------------------------------------- name matching

_TITLE_CLASS = {
    "mr": "male", "mister": "male", "sir": "male", "lord": "male",
    "mrs": "female_adult", "lady": "female_adult", "madam": "female_adult",
    "madame": "female_adult", "mistress": "female_adult",
    "miss": "female_unmarried", "ms": "female_unmarried",
    "master": "child",
    "dr": "none", "doctor": "none", "captain": "none", "colonel": "none",
}  # fmt: skip
_PUNCT = re.compile(r"[^\w\s'-]")


def normalize_name(name: str) -> str:
    """Case/punctuation-insensitive form: ``"Mr. Darcy"`` == ``"mr darcy"``."""
    name = name.replace("’", "'").casefold()
    return " ".join(_PUNCT.sub(" ", name).split())


def _split_title(norm: str) -> tuple[str | None, str]:
    """(title class or None, rest) for a normalized name; a lone title is not stripped."""
    toks = norm.split()
    if len(toks) > 1 and toks[0] in _TITLE_CLASS:
        return _TITLE_CLASS[toks[0]], " ".join(toks[1:])
    return None, norm


def _compatible(a: str | None, b: str | None) -> bool:
    """Honorific guard: two gendered/generational titles must agree (Mr. != Mrs.)."""
    if a in (None, "none") or b in (None, "none"):
        return True
    return a == b


def match_strength(predicted: set[str], gold: set[str]) -> int:
    """2 = some name matches exactly (normalized), 1 = honorific-stripped match with
    compatible titles, 0 = no match. Inputs are NORMALIZED name sets."""
    if predicted & gold:
        return 2
    for p in predicted:
        p_title, p_rest = _split_title(p)
        for g in gold:
            g_title, g_rest = _split_title(g)
            if p_rest and p_rest == g_rest and _compatible(p_title, g_title):
                return 1
    return 0


class GoldSpeaker(BaseModel):
    key: str  # normalized canonical name — the identity
    name: str  # display canonical name (first seen)
    names: set[str]  # normalized canonical + aliases


def gold_speakers(rows: list[GoldQuote]) -> dict[str, GoldSpeaker]:
    """Gold speakers keyed by normalized canonical name; aliases unioned across rows."""
    out: dict[str, GoldSpeaker] = {}
    for r in rows:
        key = normalize_name(r.speaker)
        sp = out.setdefault(key, GoldSpeaker(key=key, name=r.speaker, names={key}))
        sp.names.update(normalize_name(a) for a in r.aliases if a.strip())
    return out


class GoldConflict(BaseModel):
    """Two gold speakers whose name sets match each other — very likely ONE person left
    unmerged in Review (or a shared alias like ``Miss Bennet``). Any arm character naming
    either one becomes ambiguous and scores wrong, so this must be fixed in the gold."""

    speakers: list[str]  # the two gold display names
    shared: list[str]  # the normalized names (or honorific-stripped rests) that collide


def gold_conflicts(speakers: dict[str, GoldSpeaker]) -> list[GoldConflict]:
    """Every pair of gold speakers whose name sets match (pure; ``match_strength`` > 0)."""
    out: list[GoldConflict] = []
    keys = sorted(speakers)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            shared = _collisions(speakers[a].names, speakers[b].names)
            if shared:
                out.append(
                    GoldConflict(
                        speakers=[speakers[a].name, speakers[b].name], shared=sorted(shared)
                    )
                )
    return out


def _collisions(a: set[str], b: set[str]) -> set[str]:
    """The names behind ``match_strength(a, b) > 0``: shared exact names, else the
    honorific-stripped rests that match under compatible titles."""
    if a & b:
        return a & b
    out: set[str] = set()
    for x in a:
        x_title, x_rest = _split_title(x)
        for y in b:
            y_title, y_rest = _split_title(y)
            if x_rest and x_rest == y_rest and _compatible(x_title, y_title):
                out.add(x_rest)
    return out


def describe_conflict(c: GoldConflict) -> str:
    return (
        f"gold speakers {c.speakers[0]!r} and {c.speakers[1]!r} share name(s) "
        f"{', '.join(c.shared)} - if they are one person, merge them in Review and re-export; "
        "until then any character naming either scores as ambiguous (wrong)"
    )


class CharacterMapping(BaseModel):
    """How one predicted registry character maps onto the gold speakers."""

    character_id: str
    name: str
    gold_key: str | None = None  # None: unmapped or ambiguous
    strength: int = 0
    ambiguous_with: list[str] = []


def map_characters(
    report: AttributionReport, speakers: dict[str, GoldSpeaker]
) -> dict[str, CharacterMapping]:
    """Map every registry character to at most ONE gold speaker (best strength wins; a tie
    across distinct gold speakers is ambiguous and maps to nothing)."""
    out: dict[str, CharacterMapping] = {}
    for char in report.registry.characters:
        names = {normalize_name(n) for n in (char.canonical_name, *char.aliases) if n.strip()}
        scored = [(match_strength(names, sp.names), key) for key, sp in speakers.items()]
        best = max((s for s, _ in scored), default=0)
        winners = sorted(key for s, key in scored if s == best and best > 0)
        mapping = CharacterMapping(character_id=char.id, name=char.canonical_name, strength=best)
        if len(winners) == 1:
            mapping.gold_key = winners[0]
        elif len(winners) > 1:
            mapping.ambiguous_with = [speakers[k].name for k in winners]
        out[char.id] = mapping
    return out


# --------------------------------------------------------------------------- alignment


def align_quote(
    block_segments: list[Segment], quote_index: int, text: str, occurrence: int = 0
) -> Segment | None:
    """The segment carrying a block's ``quote_index``-th quoted span.

    Every quoted span becomes exactly ONE segment (dialogue, or narration when degraded) and
    manual edits never split segments, so the ordinal among quote-glyph-led segments is the
    fast path — confirmed by text. Otherwise fall back to an exact text match (the
    ``occurrence``-th one, for a block repeating the same quote), then a segment CONTAINING
    the quote (a flagged block's whole-block fallback), else None (unaligned).
    """
    want = text.strip()
    quoted = [s for s in block_segments if is_quoted_span(s.text)]
    if quote_index < len(quoted) and quoted[quote_index].text.strip() == want:
        return quoted[quote_index]
    same = [s for s in block_segments if s.text.strip() == want]
    if same:
        return same[min(occurrence, len(same) - 1)]
    # A FLAGGED block falls back to ONE whole-block narration segment: the quote lives inside
    # it, so it aligns there (and scores as unattributed) rather than as stale gold.
    containing = [s for s in block_segments if want and want in s.text]
    if containing:
        return containing[0]
    return None


def _segments_by_block(report: AttributionReport) -> dict[str, list[Segment]]:
    out: dict[str, list[Segment]] = defaultdict(list)
    for chapter in report.chapters:
        for seg in chapter.segments:
            out[seg.block_id].append(seg)
    return out


def _occurrences(rows: list[GoldQuote]) -> list[int]:
    """Per row: how many earlier rows in the same block share its text (realignment)."""
    seen: Counter[tuple[str, str]] = Counter()
    out: list[int] = []
    for r in rows:
        k = (r.block_id, r.text.strip())
        out.append(seen[k])
        seen[k] += 1
    return out


# ----------------------------------------------------------------------------- scoring


class Bucket(BaseModel):
    n: int = 0
    correct: int = 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def accuracy(self) -> float | None:
        return round(self.correct / self.n, 4) if self.n else None

    def add(self, ok: bool) -> None:
        self.n += 1
        self.correct += int(ok)


class QuoteOutcome(BaseModel):
    block_id: str
    quote_index: int
    chapter_index: int
    gold: str  # gold display name
    predicted: str  # gold display name it mapped to, or a <marker> / "Name (unmapped)"
    predicted_character_id: str | None = None
    correct: bool
    quote_type: str


class DriftEntry(BaseModel):
    gold_speaker: str
    predicted_characters: list[str]  # "Name [id]" of every distinct predicted char mapped here


class ConfusionPair(BaseModel):
    gold: str
    predicted: str
    count: int


class ArmScore(BaseModel):
    overall: Bucket
    unattributed: int  # gold quotes the arm left as narration
    unaligned: int  # gold quotes with no matching segment (stale gold / different book)
    by_quote_type: dict[str, Bucket]
    by_chapter: dict[int, Bucket]
    by_speaker: dict[str, Bucket]
    alias_drift: int  # sum over gold speakers of (distinct predicted chars mapped to it) - 1
    drift: list[DriftEntry]
    # Predicted characters that matched 2+ gold speakers equally (scored wrong). When the
    # arm merged what gold keeps apart (e.g. Elizabeth + alias Lizzy vs gold "Elizabeth" and
    # "Lizzy"), the GOLD likely has an unmerged duplicate — the report says so loudly.
    ambiguous: list[CharacterMapping] = []
    gold_conflicts: list[GoldConflict] = []  # gold speakers whose names collide (see above)
    confusions: list[ConfusionPair]  # top mistakes, most frequent first
    mappings: list[CharacterMapping]
    outcomes: list[QuoteOutcome]


def score_report(
    gold: list[GoldQuote], report: AttributionReport, *, top_confusions: int = 10
) -> ArmScore:
    """Score one arm's report against gold quote labels (pure)."""
    speakers = gold_speakers(gold)
    mapping = map_characters(report, speakers)
    by_block = _segments_by_block(report)

    overall = Bucket()
    by_type: dict[str, Bucket] = defaultdict(Bucket)
    by_chapter: dict[int, Bucket] = defaultdict(Bucket)
    by_speaker: dict[str, Bucket] = defaultdict(Bucket)
    confusions: Counter[tuple[str, str]] = Counter()
    drift_sets: dict[str, set[str]] = defaultdict(set)
    outcomes: list[QuoteOutcome] = []
    unattributed = unaligned = 0

    for row, occurrence in zip(gold, _occurrences(gold), strict=True):
        gold_sp = speakers[normalize_name(row.speaker)]
        seg = align_quote(by_block.get(row.block_id, []), row.quote_index, row.text, occurrence)
        char_id: str | None = None
        ok = False
        if seg is None:
            unaligned += 1
            label = UNALIGNED
        elif seg.type is SegmentType.NARRATION or seg.speaker is None:
            unattributed += 1
            label = UNATTRIBUTED
        else:
            char_id = seg.speaker
            m = mapping.get(char_id)
            if m is None:
                label = f"{char_id} (not in registry)"
            elif m.gold_key is not None:
                label = speakers[m.gold_key].name
                drift_sets[m.gold_key].add(char_id)
                ok = m.gold_key == gold_sp.key
            elif m.ambiguous_with:
                label = f"{m.name} (ambiguous)"
            else:
                label = f"{m.name} (unmapped)"
        qtype = row.quote_type.value if row.quote_type else UNLABELED_TYPE
        overall.add(ok)
        by_type[qtype].add(ok)
        by_chapter[row.chapter_index].add(ok)
        by_speaker[gold_sp.name].add(ok)
        if not ok:
            confusions[(gold_sp.name, label)] += 1
        outcomes.append(
            QuoteOutcome(
                block_id=row.block_id,
                quote_index=row.quote_index,
                chapter_index=row.chapter_index,
                gold=gold_sp.name,
                predicted=label,
                predicted_character_id=char_id,
                correct=ok,
                quote_type=qtype,
            )
        )

    drift = [
        DriftEntry(
            gold_speaker=speakers[key].name,
            predicted_characters=sorted(f"{mapping[c].name} [{c}]" for c in ids),
        )
        for key, ids in sorted(drift_sets.items())
        if len(ids) > 1
    ]
    return ArmScore(
        overall=overall,
        unattributed=unattributed,
        unaligned=unaligned,
        by_quote_type=dict(sorted(by_type.items())),
        by_chapter=dict(sorted(by_chapter.items())),
        by_speaker=dict(sorted(by_speaker.items(), key=lambda kv: (-kv[1].n, kv[0]))),
        alias_drift=sum(len(ids) - 1 for ids in drift_sets.values()),
        drift=drift,
        ambiguous=sorted(
            (m for m in mapping.values() if m.ambiguous_with), key=lambda m: m.character_id
        ),
        gold_conflicts=gold_conflicts(speakers),
        confusions=[
            ConfusionPair(gold=g, predicted=p, count=n)
            for (g, p), n in sorted(confusions.items(), key=lambda kv: (-kv[1], kv[0]))[
                :top_confusions
            ]
        ],
        mappings=sorted(mapping.values(), key=lambda m: m.character_id),
        outcomes=outcomes,
    )


# ------------------------------------------------------------------------- gold export


def splitter_convention(book: NormalizedBook) -> DialogueConvention:
    """The convention the pipeline splits on — detected over the FULL book, exactly as
    ``attribute_book`` does, so gold quote ordinals match the attributed spans."""
    detection = detect_dialogue_convention(
        "\n".join(
            b.text for ch in book.chapters for b in ch.blocks if b.type is BlockType.PARAGRAPH
        )
    )
    if detection.convention is DialogueConvention.SINGLE_CURLY:
        return DialogueConvention.SINGLE_CURLY
    return DialogueConvention.DOUBLE


class GoldExport(BaseModel):
    rows: list[GoldQuote]
    skipped_unattributed: int  # quotes left as narration in the effective report
    skipped_unaligned: int  # quotes with no segment (chapter not attributed / stale report)
    conflicts: list[GoldConflict] = []  # likely-unmerged duplicate speakers: warn the user


def export_gold(
    book: NormalizedBook, report: AttributionReport, chapters: tuple[int, ...] = ()
) -> GoldExport:
    """Turn an EFFECTIVE report (raw + Review edits) into gold rows, one per quoted span.

    Quotes come from the book's own deterministic split (authoritative offsets); each is
    labeled with the speaker of its aligned segment, named by the report's registry
    (canonical + aliases). Quotes still unattributed are skipped — there is no gold speaker
    to record, and silently guessing one would poison the benchmark.
    """
    wanted = set(chapters)
    unknown = wanted - set(range(1, len(book.chapters) + 1))
    if unknown:
        raise ValueError(
            f"{book.book_meta.book_id}: no such chapter(s) {sorted(unknown)} "
            f"(book has {len(book.chapters)})"
        )
    convention = splitter_convention(book)
    by_block = _segments_by_block(report)
    rows: list[GoldQuote] = []
    skipped_unattributed = skipped_unaligned = 0
    for ci, chapter in enumerate(book.chapters, start=1):
        if wanted and ci not in wanted:
            continue
        for block in chapter.blocks:
            if block.type is not BlockType.PARAGRAPH:
                continue
            spans = thought_candidate_spans(block.id, block.text, [], convention)
            seen: Counter[str] = Counter()
            for ordinal, span in quoted_ordinals(spans):
                occurrence = seen[span.text.strip()]
                seen[span.text.strip()] += 1
                seg = align_quote(by_block.get(block.id, []), ordinal, span.text, occurrence)
                if seg is None:
                    skipped_unaligned += 1
                    continue
                if seg.type is SegmentType.NARRATION or seg.speaker is None:
                    skipped_unattributed += 1
                    continue
                char = report.registry.get(seg.speaker)
                name = char.canonical_name if char else seg.speaker
                rows.append(
                    GoldQuote(
                        book_id=book.book_meta.book_id,
                        chapter_index=ci,
                        block_id=block.id,
                        quote_index=ordinal,
                        char_start=span.start,
                        char_end=span.start + len(span.text),
                        text=span.text,
                        speaker=name,
                        aliases=list(char.aliases) if char else [],
                    )
                )
    return GoldExport(
        rows=rows,
        skipped_unattributed=skipped_unattributed,
        skipped_unaligned=skipped_unaligned,
        conflicts=gold_conflicts(gold_speakers(rows)),
    )
