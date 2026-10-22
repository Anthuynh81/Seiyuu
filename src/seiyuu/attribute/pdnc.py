"""Import PDNC speaker labels as gold quote rows for the accuracy harness.

PDNC (the Project Dialogism Novel Corpus, Vishnubhotla et al., LREC 2022, CC-BY 4.0) labels
every quotation of a novel with its speaker. Its text is NOT our ingest: whitespace, line
breaks, quote glyphs, ``_italic_`` markup, front matter and even drop-cap letters differ, so
quotes are matched by TEXT, never by PDNC's byte offsets.

Matching is a monotonic three-pass alignment over the two quote sequences in reading order:

1. **Anchors.** Every PDNC sub-quotation and every quoted span of the book is folded to a
   space-free word key (case, quote glyphs, apostrophes, punctuation, whitespace and
   ``_`` all dropped, so ``any thing`` == ``anything``). ``difflib.SequenceMatcher`` aligns
   the two key sequences; equal keys inside its matching blocks pair 1:1. Because the
   alignment is monotonic, a repeated short quote ("Yes.") pairs with the occurrence at the
   same place in the story, not the first one.
2. **Gaps.** Between consecutive anchors, the leftover sub-quotes and spans are re-aligned at
   the TOKEN level (again monotonic; only runs of 2+ tokens count). This recovers split
   differences (one PDNC sub-quote covering two of our spans, two PDNC sub-quotes inside one
   of our spans) and small wording differences in long quotes.
3. **Near-identical.** A sub-quote and a span still left over between the same confirmed
   neighbours pair 1:1 when their keys are >= ``MIN_SIMILARITY`` alike at the character
   level (edition spelling: mama/mamma, teazed/teased, a lost drop-cap letter).

Passes 1-2 label a span only when the match is confident from both sides: at least
``MIN_COVERAGE`` of the span's tokens align to sub-quotes of ONE speaker, NO token aligns to
another speaker, and every such sub-quote has at least ``MIN_COVERAGE`` of its own tokens
aligned to the book. An anchor whose key repeats un-anchored nearby (a second book "Lizzy!",
another speaker's "Yes.") is ambiguous: it and its rivals are blocked from every pass. Pass
3 pairs only unique, single-speaker candidates. Anything weaker stays unlabeled and is
reported, never guessed.

Pure apart from the explicit CSV loaders; the loaders parse the untrusted CSV with ``csv`` and
``ast.literal_eval`` only.
"""

import ast
import csv
import re
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from itertools import pairwise
from pathlib import Path

from pydantic import BaseModel

from seiyuu.attribute.evaluate import (
    BookQuote,
    GoldConflict,
    GoldQuote,
    QuoteType,
    _split_title,
    book_quotes,
    check_chapters,
    gold_conflicts,
    gold_speakers,
    normalize_name,
)
from seiyuu.ingest.models import NormalizedBook

MIN_COVERAGE = 0.9
# Token-level gap alignment is quadratic in the worst case: a gap bigger than this (a whole
# missing chapter, a different book) is left unmatched rather than stalling the import.
MAX_GAP_TOKENS = 20_000
MIN_TOKEN_RUN = 2  # shortest token run that counts as gap evidence
MIN_SIMILARITY = 0.9  # char-level ratio for the near-identical 1:1 pass
MIN_SIMILAR_CHARS = 8  # shorter keys only ever match exactly (as anchors)

_APOSTROPHES = re.compile(r"['’‘`]")
_WORD = re.compile(r"[^\W_]+")


def fold_tokens(text: str) -> list[str]:
    """Word tokens for matching: case/glyph/punctuation/whitespace/underscore-insensitive.
    Apostrophes are deleted (``don’t`` == ``dont``) rather than split on."""
    return _WORD.findall(_APOSTROPHES.sub("", text.casefold()))


# ----------------------------------------------------------------------------- loading


class PdncQuote(BaseModel):
    quote_id: str
    speaker: str
    quote_type: QuoteType | None = None
    sub_quotes: list[str]  # the quote's pieces, split by intervening narration


class PdncCharacter(BaseModel):
    name: str  # PDNC "Main Name" — what quotation_info.csv's ``speaker`` column holds
    aliases: list[str] = []


def _literal(value: str, where: str) -> object:
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"{where}: not a python literal: {value[:80]!r}") from exc


def _read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing column(s) {sorted(missing)}")
        return list(reader)


def load_pdnc_quotes(path: Path) -> list[PdncQuote]:
    """quotation_info.csv -> quotes, in the file's (reading) order."""
    out: list[PdncQuote] = []
    rows = _read_csv(path, {"quoteID", "subQuotationList", "speaker", "quoteType"})
    for lineno, row in enumerate(rows, start=2):
        where = f"{path}:{lineno} ({row['quoteID']})"
        subs = _literal(row["subQuotationList"], where)
        if not isinstance(subs, (list, tuple)) or not all(isinstance(s, str) for s in subs):
            raise ValueError(f"{where}: subQuotationList must be a list of strings")
        speaker = row["speaker"].strip()
        if not speaker:
            raise ValueError(f"{where}: empty speaker")
        qtype = row["quoteType"].strip().casefold()
        out.append(
            PdncQuote(
                quote_id=row["quoteID"],
                speaker=speaker,
                quote_type=QuoteType(qtype) if qtype in set(QuoteType) else None,
                sub_quotes=list(subs),
            )
        )
    return out


def load_pdnc_characters(path: Path) -> dict[str, PdncCharacter]:
    """character_info.csv -> characters keyed by main name."""
    out: dict[str, PdncCharacter] = {}
    rows = _read_csv(path, {"Main Name", "Aliases"})
    for lineno, row in enumerate(rows, start=2):
        name = row["Main Name"].strip()
        if not name:
            continue
        raw = row["Aliases"].strip()
        aliases = _literal(raw, f"{path}:{lineno} ({name})") if raw else []
        if isinstance(aliases, str):
            aliases = [aliases]
        if not isinstance(aliases, (list, tuple, set, frozenset)):
            raise ValueError(f"{path}:{lineno} ({name}): Aliases must be a list or set")
        out[name] = PdncCharacter(
            name=name, aliases=sorted({str(a).strip() for a in aliases} - {""})
        )
    return out


# --------------------------------------------------------------------------- alignment


def _anchor_pairs(p_keys: list[str], s_keys: list[str]) -> list[tuple[int, int]]:
    """1:1 pairs of identical non-empty keys from a monotonic sequence alignment."""
    matcher = SequenceMatcher(None, p_keys, s_keys, autojunk=False)
    return [
        (blk.a + k, blk.b + k)
        for blk in matcher.get_matching_blocks()
        for k in range(blk.size)
        if p_keys[blk.a + k]
    ]


def _between(sorted_items: list[int], lo: int, hi: int) -> list[int]:
    """The items strictly between ``lo`` and ``hi``."""
    return sorted_items[bisect_right(sorted_items, lo) : bisect_left(sorted_items, hi)]


def _unique_anchors(
    anchors: list[tuple[int, int]], p_keys: list[str], s_keys: list[str], labels: list[str]
) -> tuple[list[tuple[int, int]], set[int], set[int]]:
    """Drop anchors that only LOOK confident because their key repeats nearby.

    An anchor's window runs from the nearest anchor with a DIFFERENT key before it to the
    nearest one after it (a run of equal-key anchors shares one window). The run is
    ambiguous when its window also holds an un-anchored span with the same key (which of
    the book's two "Lizzy!"s did PDNC annotate?) or an un-anchored sub-quote with the same
    key but another speaker (whose "Yes." is this one?). Ambiguous anchors and their
    rivals come back blocked: no later pass may label them either.

    -> (kept anchors, blocked sub-quotes, blocked spans)
    """
    anchored_p = {p for p, _ in anchors}
    anchored_s = {s for _, s in anchors}
    free_p: dict[str, list[int]] = defaultdict(list)
    free_s: dict[str, list[int]] = defaultdict(list)
    for p, key in enumerate(p_keys):
        if key and p not in anchored_p:
            free_p[key].append(p)
    for s, key in enumerate(s_keys):
        if key and s not in anchored_s:
            free_s[key].append(s)
    end = (len(p_keys), len(s_keys))
    kept: list[tuple[int, int]] = []
    blocked_p: set[int] = set()
    blocked_s: set[int] = set()
    i = 0
    while i < len(anchors):
        key = p_keys[anchors[i][0]]
        j = i
        while j < len(anchors) and p_keys[anchors[j][0]] == key:
            j += 1
        run = anchors[i:j]
        lo = anchors[i - 1] if i else (-1, -1)
        hi = anchors[j] if j < len(anchors) else end
        run_labels = {labels[p] for p, _ in run}
        rival_s = _between(free_s.get(key, []), lo[1], hi[1])
        rival_p = [
            p for p in _between(free_p.get(key, []), lo[0], hi[0]) if run_labels != {labels[p]}
        ]
        if rival_s or rival_p:
            blocked_p.update(p for p, _ in run)
            blocked_p.update(rival_p)
            blocked_s.update(s for _, s in run)
            blocked_s.update(rival_s)
        else:
            kept.extend(run)
        i = j
    return kept, blocked_p, blocked_s


def _gap_token_pairs(
    p_items: list[int], s_items: list[int], p_toks: list[list[str]], s_toks: list[list[str]]
) -> list[tuple[int, int]]:
    """(sub-quote, span) for every token aligned between two gap regions. Only runs of
    ``MIN_TOKEN_RUN``+ tokens count: a lone common word ("yes", "my") shared by unrelated
    text is not evidence."""
    p_stream = [(i, t) for i in p_items for t in p_toks[i]]
    s_stream = [(j, t) for j in s_items for t in s_toks[j]]
    if not p_stream or not s_stream or max(len(p_stream), len(s_stream)) > MAX_GAP_TOKENS:
        return []
    matcher = SequenceMatcher(
        None, [t for _, t in p_stream], [t for _, t in s_stream], autojunk=False
    )
    return [
        (p_stream[blk.a + k][0], s_stream[blk.b + k][0])
        for blk in matcher.get_matching_blocks()
        if blk.size >= MIN_TOKEN_RUN
        for k in range(blk.size)
    ]


def _speakers_in(cover: Counter[tuple[int, int]], labels: list[str]) -> dict[int, set[str]]:
    """span -> every speaker with at least one token aligned into it."""
    out: dict[int, set[str]] = defaultdict(set)
    for p, s in cover:
        out[s].add(labels[p])
    return out


def _confident(
    cover: Counter[tuple[int, int]],
    labels: list[str],
    p_len: list[int],
    s_len: list[int],
) -> dict[int, list[int]]:
    """Spans whose aligned tokens all come from ONE speaker's sub-quotes and cover
    >= MIN_COVERAGE of the span, each such sub-quote itself >= MIN_COVERAGE aligned to the
    book -> {span: [sub-quotes]}. Any aligned token of a second speaker vetoes the span: a
    span holding two speakers' words has no single gold speaker, however small the share."""
    p_covered: Counter[int] = Counter()
    by_span: dict[int, Counter[str]] = defaultdict(Counter)  # span -> speaker -> tokens
    feeders: dict[int, set[int]] = defaultdict(set)
    for (p, s), n in cover.items():
        p_covered[p] += n
        by_span[s][labels[p]] += n
        feeders[s].add(p)
    out: dict[int, list[int]] = {}
    for s, per_label in by_span.items():
        if len(per_label) != 1:
            continue
        (n,) = per_label.values()
        if n < MIN_COVERAGE * s_len[s]:
            continue
        subs = sorted(feeders[s])
        if all(p_covered[p] >= MIN_COVERAGE * p_len[p] for p in subs):
            out[s] = subs
    return out


def _similar_pairs(
    p_keys: list[str],
    s_keys: list[str],
    labels: list[str],
    confirmed: dict[int, list[int]],
    speakers_in: dict[int, set[str]],
    blocked_p: set[int],
    blocked_s: set[int],
) -> dict[int, list[int]]:
    """Last pass: a leftover sub-quote and a leftover span between the same confirmed
    neighbours pair 1:1 when their space-free keys are near-identical (spelling variants
    such as mama/mamma, teazed/teased, a lost drop-cap letter).

    Only UNIQUE pairs count (the sub-quote has exactly one such span in its window and the
    span exactly one such sub-quote), and never a span that already holds another speaker's
    aligned tokens: "X, oh dear me!" stays unlabeled when "Oh dear me!" is a second
    speaker's line, even though X alone is >= MIN_SIMILARITY alike."""
    pairs = sorted((subs[0], s) for s, subs in confirmed.items())  # monotonic by sub
    used_p = {p for subs in confirmed.values() for p in subs}
    free_s = [
        s
        for s in range(len(s_keys))
        if s not in confirmed and s not in blocked_s and len(s_keys[s]) >= MIN_SIMILAR_CHARS
    ]
    found: list[tuple[int, int]] = []  # ordered by sub-quote
    for p, key in enumerate(p_keys):
        if p in used_p or p in blocked_p or len(key) < MIN_SIMILAR_CHARS:
            continue
        k = bisect_left(pairs, (p, -1))
        lo = pairs[k - 1][1] if k else -1
        hi = pairs[k][1] if k < len(pairs) else len(s_keys)
        for s in _between(free_s, lo, hi):
            if speakers_in.get(s, set()) - {labels[p]}:
                continue
            matcher = SequenceMatcher(None, key, s_keys[s], autojunk=False)
            if matcher.quick_ratio() >= MIN_SIMILARITY and matcher.ratio() >= MIN_SIMILARITY:
                found.append((p, s))
    per_p = Counter(p for p, _ in found)
    per_s = Counter(s for _, s in found)
    out: dict[int, list[int]] = {}
    last = -1
    for p, s in found:
        if per_p[p] == 1 and per_s[s] == 1 and s > last:
            out[s] = [p]
            last = s
    return out


def align_quotes(
    sub_quotes: list[str], labels: list[str], spans: list[str]
) -> dict[int, list[int]]:
    """Map book spans to PDNC sub-quotations (pure).

    ``sub_quotes`` is every PDNC sub-quotation in reading order and ``labels`` its speaker;
    ``spans`` is every quoted span of the book in reading order. Returns
    ``{span_index: [sub_quote_index, ...]}`` for the CONFIDENT matches only (see the module
    docstring); a span's sub-quotes always share one speaker.
    """
    p_toks = [fold_tokens(t) for t in sub_quotes]
    s_toks = [fold_tokens(t) for t in spans]
    p_keys = ["".join(t) for t in p_toks]  # space-free: "any thing" == "anything"
    s_keys = ["".join(t) for t in s_toks]

    anchors, blocked_p, blocked_s = _unique_anchors(
        _anchor_pairs(p_keys, s_keys), p_keys, s_keys, labels
    )
    # token-level coverage: cover[(p, s)] = tokens of sub-quote p aligned into span s
    cover: Counter[tuple[int, int]] = Counter()
    p_len = [len(t) for t in p_toks]
    for p, s in anchors:
        # equal keys: credit the span's token count to both sides, so a sub-quote whose
        # words merely split differently ("any thing") still counts as fully covered
        cover[(p, s)] += len(s_toks[s])
        p_len[p] = len(s_toks[s])
    bounds = [(-1, -1), *anchors, (len(p_toks), len(s_toks))]
    for (pa, sa), (pb, sb) in pairwise(bounds):
        if pb - pa > 1 and sb - sa > 1:
            gap_p = [p for p in range(pa + 1, pb) if p not in blocked_p]
            gap_s = [s for s in range(sa + 1, sb) if s not in blocked_s]
            for p, s in _gap_token_pairs(gap_p, gap_s, p_toks, s_toks):
                cover[(p, s)] += 1
    confirmed = _confident(cover, labels, p_len, [len(t) for t in s_toks])
    return confirmed | _similar_pairs(
        p_keys, s_keys, labels, confirmed, _speakers_in(cover, labels), blocked_p, blocked_s
    )


# ------------------------------------------------------------------------------ import


class ChapterCoverage(BaseModel):
    chapter_index: int
    title: str
    spans: int  # our quoted spans in the chapter
    labeled: int  # of those, labeled from PDNC
    pdnc_quotes: int  # distinct PDNC quotes that landed in the chapter


class UnmatchedQuote(BaseModel):
    quote_id: str
    speaker: str
    text: str  # sub-quotes joined with " / "
    partial: bool = False  # some sub-quotes matched, others did not
    # where it would sit: the chapters of the nearest matched sub-quotes before and after it
    # (one chapter, two at a chapter boundary, none when nothing matched)
    near_chapters: list[int] = []


class UnlabeledSpan(BaseModel):
    chapter_index: int
    block_id: str
    quote_index: int
    text: str


class DroppedAlias(BaseModel):
    character: str
    alias: str
    reason: str


class PdncImport(BaseModel):
    rows: list[GoldQuote]
    chapters_chosen: bool = False  # --chapters given: rows/spans/lists are per chapters
    # WHOLE-BOOK counts (the alignment always runs over the whole book):
    pdnc_quotes: int
    pdnc_quotes_matched: int  # at least one span labeled from the quote
    pdnc_quotes_fully_matched: int  # every sub-quote matched
    pdnc_sub_quotes: int
    pdnc_sub_quotes_matched: int
    # chosen-chapter counts:
    pdnc_quotes_in_chapters: int  # matched quotes landing in the chosen chapters
    spans: int  # our quoted spans in the chosen chapters
    spans_labeled: int
    chapters: list[ChapterCoverage]
    # PDNC quotes with at least one unmatched sub-quote (with chapters chosen: only those
    # with a nearest matched neighbour in a chosen chapter)
    unmatched: list[UnmatchedQuote]
    unlabeled: list[UnlabeledSpan]  # our spans (chosen chapters) with no PDNC label
    unknown_speakers: list[str] = []  # speakers absent from character_info (no aliases)
    dropped_aliases: list[DroppedAlias] = []
    conflicts: list[GoldConflict] = []


def _speaker_names(
    characters: dict[str, PdncCharacter],
) -> tuple[dict[str, list[str]], list[DroppedAlias]]:
    """Aliases per main name (main name included). Two kinds of alias are dropped, because
    each makes an ordinary predicted character tie between two gold speakers (score wrong):

    - an alias that is ANOTHER character's main name (PDNC lists 'Lady Lucas' under
      Charlotte Lucas);
    - a bare, untitled surname when ANOTHER character carries it with a male title (PDNC
      lists 'Bennet' under Mrs. Bennet while 'Mr. Bennet' is his own character): a
      predicted {Mr. Bennet, Bennet} would match both exactly. In these novels a bare
      surname names the Mr., so 'Darcy' under Mr. Darcy (no other 'Mr. Darcy') is kept.
    """
    mains = {normalize_name(n): n for n in characters}
    male_rests: dict[str, set[str]] = defaultdict(set)  # bare surname -> its Mr. bearers
    for name, char in characters.items():
        for n in (name, *char.aliases):
            title, rest = _split_title(normalize_name(n))
            if title == "male":
                male_rests[rest].add(name)
    out: dict[str, list[str]] = {}
    dropped: list[DroppedAlias] = []
    for name, char in characters.items():
        keep: set[str] = {name}
        for alias in char.aliases:
            norm = normalize_name(alias)
            if norm == normalize_name(name):
                continue  # the main name itself: always kept
            other = mains.get(norm)
            title, _ = _split_title(norm)
            bearers = sorted(male_rests.get(norm, set()) - {name}) if title is None else []
            if other is not None and other != name:
                reason = f"main name of character {other!r}"
            elif bearers:
                reason = f"bare surname of the male-titled character {bearers[0]!r}"
            else:
                keep.add(alias)
                continue
            dropped.append(DroppedAlias(character=name, alias=alias, reason=reason))
        out[name] = sorted(keep)
    return out, dropped


def import_pdnc(
    book: NormalizedBook,
    quotes: list[PdncQuote],
    characters: dict[str, PdncCharacter],
    chapters: tuple[int, ...] = (),
) -> PdncImport:
    """PDNC quotes -> gold rows on the book's own quoted spans (one row per labeled span).

    The alignment always runs over the WHOLE book (more anchors, right occurrences);
    ``chapters`` only restricts which rows and coverage counts are reported.
    """
    check_chapters(book, chapters)
    every = book_quotes(book)
    sub_owner = [qi for qi, q in enumerate(quotes) for _ in q.sub_quotes]
    matches = align_quotes(
        [text for q in quotes for text in q.sub_quotes],
        [quotes[qi].speaker for qi in sub_owner],
        [bq.span.text for bq in every],
    )
    names, dropped = _speaker_names(characters)

    matched_subs: set[int] = {p for subs in matches.values() for p in subs}
    subs_of: dict[int, list[int]] = defaultdict(list)
    for p, qi in enumerate(sub_owner):
        subs_of[qi].append(p)

    wanted = set(chapters) or set(range(1, len(book.chapters) + 1))
    rows: list[GoldQuote] = []
    unlabeled: list[UnlabeledSpan] = []
    per_chapter: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    quotes_per_chapter: dict[int, set[int]] = defaultdict(set)
    unknown: set[str] = set()
    for s, bq in enumerate(every):
        if bq.chapter_index not in wanted:
            continue
        per_chapter[bq.chapter_index][0] += 1
        subs = matches.get(s)
        if subs is None:
            unlabeled.append(_unlabeled(bq))
            continue
        owners = sorted({sub_owner[p] for p in subs})
        quote = quotes[owners[0]]  # several only when one span joins same-speaker quotes
        types = {quotes[qi].quote_type for qi in owners}
        per_chapter[bq.chapter_index][1] += 1
        quotes_per_chapter[bq.chapter_index].update(owners)
        if quote.speaker not in names:
            unknown.add(quote.speaker)
        rows.append(
            GoldQuote(
                book_id=book.book_meta.book_id,
                chapter_index=bq.chapter_index,
                block_id=bq.block_id,
                quote_index=bq.quote_index,
                char_start=bq.span.start,
                char_end=bq.span.start + len(bq.span.text),
                text=bq.span.text,
                speaker=quote.speaker,
                aliases=names.get(quote.speaker, []),
                quote_type=types.pop() if len(types) == 1 else None,
            )
        )

    chapter_near = _chapter_near(len(sub_owner), matches, every)
    unmatched: list[UnmatchedQuote] = []
    fully = any_matched = 0
    for qi, q in enumerate(quotes):
        hit = [p in matched_subs for p in subs_of[qi]]
        any_matched += any(hit)
        if all(hit) and hit:
            fully += 1
            continue
        near = chapter_near[subs_of[qi][0]] if subs_of[qi] else []
        if chapters and not wanted.intersection(near):
            continue  # the whole-book counts include it; the list is per chosen chapters
        unmatched.append(
            UnmatchedQuote(
                quote_id=q.quote_id,
                speaker=q.speaker,
                text=" / ".join(" ".join(t.split()) for t in q.sub_quotes),
                partial=any(hit),
                near_chapters=near,
            )
        )

    return PdncImport(
        rows=rows,
        chapters_chosen=bool(chapters),
        pdnc_quotes=len(quotes),
        pdnc_quotes_matched=any_matched,
        pdnc_quotes_fully_matched=fully,
        pdnc_sub_quotes=len(sub_owner),
        pdnc_sub_quotes_matched=len(matched_subs),
        pdnc_quotes_in_chapters=len({q for qs in quotes_per_chapter.values() for q in qs}),
        spans=sum(n for n, _ in per_chapter.values()),
        spans_labeled=len(rows),
        chapters=[
            ChapterCoverage(
                chapter_index=ci,
                title=book.chapters[ci - 1].title,
                spans=per_chapter[ci][0],
                labeled=per_chapter[ci][1],
                pdnc_quotes=len(quotes_per_chapter[ci]),
            )
            for ci in sorted(wanted)
        ],
        unmatched=unmatched,
        unlabeled=unlabeled,
        unknown_speakers=sorted(unknown),
        dropped_aliases=dropped,
        conflicts=gold_conflicts(gold_speakers(rows)),
    )


def _chapter_near(
    n_subs: int, matches: dict[int, list[int]], every: list[BookQuote]
) -> list[list[int]]:
    """Per sub-quote: the chapters of the nearest matched sub-quote before and after it
    (itself, when matched) - where an unmatched quote would sit in the book."""
    at = {p: every[s].chapter_index for s, subs in matches.items() for p in subs}
    before: list[int | None] = [None] * n_subs
    after: list[int | None] = [None] * n_subs
    last: int | None = None
    for p in range(n_subs):
        last = at.get(p, last)
        before[p] = last
    last = None
    for p in reversed(range(n_subs)):
        last = at.get(p, last)
        after[p] = last
    return [sorted({c for c in pair if c is not None}) for pair in zip(before, after, strict=True)]


def _unlabeled(bq: BookQuote) -> UnlabeledSpan:
    return UnlabeledSpan(
        chapter_index=bq.chapter_index,
        block_id=bq.block_id,
        quote_index=bq.quote_index,
        text=" ".join(bq.span.text.split()),
    )


def _pct(n: int, d: int) -> str:
    return f"{n}/{d} ({n / d * 100:.1f}%)" if d else f"{n}/{d}"


def _clip(text: str, width: int = 90) -> str:
    return text if len(text) <= width else text[: width - 3] + "..."


def coverage_report(result: PdncImport, *, examples: int = 5) -> list[str]:
    """The coverage report in plain words, one line per entry."""
    r = result
    scope = "the chosen chapters" if r.chapters_chosen else "all chapters"
    lines = [
        f"PDNC quotes matched to the book (whole book): "
        f"{_pct(r.pdnc_quotes_matched, r.pdnc_quotes)} "
        f"(fully, every piece: {_pct(r.pdnc_quotes_fully_matched, r.pdnc_quotes)})",
        f"PDNC quote pieces (sub-quotations) matched (whole book): "
        f"{_pct(r.pdnc_sub_quotes_matched, r.pdnc_sub_quotes)}",
        f"Our quoted spans labeled in {scope}: {_pct(r.spans_labeled, r.spans)} "
        f"(from {r.pdnc_quotes_in_chapters} PDNC quotes)",
        "Per chapter (labeled / our spans, PDNC quotes):",
    ]
    for c in r.chapters:
        lines.append(
            f"  ch {c.chapter_index:>3} {_clip(c.title, 28):<28} "
            f"{c.labeled:>4} / {c.spans:<4} {c.pdnc_quotes:>4} PDNC quotes"
        )
    if r.unmatched:
        where = " near the chosen chapters" if r.chapters_chosen else ""
        lines.append(
            f"PDNC quotes not (fully) matched{where}: {len(r.unmatched)}; first {examples} of them:"
        )
        for u in r.unmatched[:examples]:
            tag = " (partly matched)" if u.partial else ""
            near = f" ~ch {'/'.join(map(str, u.near_chapters))}" if u.near_chapters else ""
            lines.append(f"  {u.quote_id} [{u.speaker}]{near}{tag}: {_clip(u.text)}")
    if r.unlabeled:
        lines.append(f"Our spans left without a PDNC label: {len(r.unlabeled)}; first {examples}:")
        for s in r.unlabeled[:examples]:
            lines.append(f"  ch {s.chapter_index} {s.block_id} #{s.quote_index}: {_clip(s.text)}")
    if r.unknown_speakers:
        lines.append(
            "Speakers missing from the character file (labeled, but with no aliases): "
            + ", ".join(r.unknown_speakers)
        )
    for d in r.dropped_aliases:
        lines.append(f"Dropped alias {d.alias!r} of {d.character!r}: it is the {d.reason}")
    return lines
