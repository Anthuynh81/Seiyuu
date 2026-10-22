"""Pure transcript-vs-text scoring for whisper validation (no I/O, no model).

Both sides are first folded into spoken words through the normalizer's own rules (so whisper's
"Mr.", "&", "12,000", "3rd", "VIII" meet the "Mister", "and", "twelve thousand", "third", "the
eighth" the engine was given), then lowercased, stutters dropped ("P-P-Potter"), contractions
and dialect negations opened ("didn't", "dont", "nah", "nothin'") and punctuation dropped. Then:

1. **Numbers by value.** Each run of number words is parsed into the values it can mean
   ("nineteen eighty-four" = 1984, "twelve hundred" = 1200, "nine one one" = 911, "twenty-first"
   = 21, "two and a half" = 2.5, "two point five million" = 2,500,000). Number phrases whose values
   agree count as the same words, whatever their spelling; a number heard as a different number
   fails ("212" heard as "213"). A lone "one"/"two"/"four"/"eight" is only forgiven when its
   homophone ("won", "to", "for", "ate") stands in its place.
2. **Similarity.** The optimal (LCS/Indel) character similarity of the reconciled strings must
   reach ``min_ratio``. Character level, so a spelling slip ("Moreti") costs a little, not a word.
3. **Structure.** An optimal word alignment must show no run of more than ``max_word_gap`` wrong,
   missing or extra words (a dropped, looped or invented clause), no lost or added negation
   ("not", "never", "without", "unlocked" / "locked"), no words cut off the start or end (an
   autoregressive engine stopping early), and no more than ~12% of the words wrong in total.
   Words that only differ in spelling ("colour" / "color", "Her my oh nee" / "Hermione") are
   spelling differences, not missing words.

A segment can be checked against alternate expected texts (the book text without its
pronunciation respellings): the best verdict wins, since whisper writes the real name.

difflib is deliberately not used: its greedy matcher collapsed word-perfect paragraphs (autojunk)
and misaligned repeated refrains, and a ratio alone has no sense of structure.
"""

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from rapidfuzz.distance import Indel, Levenshtein

from seiyuu.normalize import normalize_text

DEFAULT_MAX_WORD_GAP = 4
ERROR_BUDGET = 0.12  # share of the expected words that may be wrong in total
MIN_ERROR_BUDGET = 3  # ...but never fewer than this many, so short lines rely on similarity
SPELLING_VARIANT_SIMILARITY = 0.6  # random phrases: p99 ~0.5; respellings/split names: 0.63+
SPELLING_VARIANT_MAX_WORDS = 8
SAME_WORD_SIMILARITY = 0.75  # one word for one: colour/color 0.91, grey/gray 0.75
MAX_GROUP = 8  # longest list of number phrases one side may read as a single phrase
MAX_READINGS = 512  # stop enumerating a group's joined readings past this many

_TOKEN = re.compile(r"[^\W_]+|[,.;:!?]")
_BOUNDARY = frozenset(",.;:!?")
# whisper glues "3pm", "10am", "2:30pm": split letters off digits (not 21st / 1990s)
_GLUED_UNIT = re.compile(r"(\d)(?!(?:st|nd|rd|th|s)\b)([A-Za-z])")
_STUTTER = re.compile(r"\b(\w{1,2})-(?=\1)")  # "p-p-potter" -> "potter", "d-don't" -> "don't"
_AP = "['’]"
_SPOKEN_FORMS = [  # applied in order to casefolded text
    (re.compile(rf"\bne{_AP}er\b"), "never"),
    (re.compile(rf"\b(?:nothin|nuthin){_AP}?(?!\w)"), "nothing"),
    (re.compile(r"\b(?:nah|naw|nay|nae|nope)\b"), "no"),
    (re.compile(r"\bdunno\b"), "do not know"),
    (re.compile(r"\bcannae\b"), "can not"),
    (re.compile(r"\binnit\b"), "is not it"),
    (re.compile(rf"\bcan{_AP}?t\b"), "can not"),
    (re.compile(r"\bcannot\b"), "can not"),
    (re.compile(rf"\bwon{_AP}t\b|\bwont\b"), "will not"),
    (re.compile(rf"\bshan{_AP}?t\b"), "shall not"),
    (re.compile(rf"\bain{_AP}?t\b"), "aint"),
    (re.compile(rf"\b(\w+)n{_AP}t\b"), r"\1 not"),
    (
        re.compile(
            r"\b(do|does|did|is|are|was|were|has|have|had|could|would|should|must|need|might)"
            r"nt\b"
        ),
        r"\1 not",
    ),
]
_NEGATIONS = frozenset(
    "not never no nobody none nothing nowhere neither nor aint without hardly barely scarcely "
    "nought naught".split()
)
_NEGATING_PREFIXES = ("un", "in", "im", "il", "ir", "dis", "non")
_FILLERS = frozenset("oh ah uh um er erm hmm mm".split())
# laughter and sighs whisper often leaves out ("Haha,", "Phew.", "Ahem")
_VOCAL_SOUND = re.compile(
    r"(?:h[aeiou]){2,}h?|he+h|a+h+|o+h+|h+m+(?:ph)?|ahem|phew|whoa|pfft|tsk|ugh|argh|ha|hah"
)

_NUMBER_KINDS: dict[str, tuple[str, Fraction]] = {}


def _vocab(kind: str, words: str, values: list[int | Fraction]) -> None:
    for word, value in zip(words.split(), values, strict=True):
        _NUMBER_KINDS[word] = (kind, Fraction(value))


_vocab("unit", "zero one two three four five six seven eight nine", list(range(10)))
_vocab("unit", "first second third fourth fifth sixth seventh eighth ninth", list(range(1, 10)))
_TEEN = "ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen"
_vocab("teen", _TEEN, list(range(10, 20)))
_vocab(
    "teen",
    "tenth eleventh twelfth thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth "
    "nineteenth",
    list(range(10, 20)),
)
_TENS = list(range(20, 100, 10))
_vocab("tens", "twenty thirty forty fifty sixty seventy eighty ninety", _TENS)
_vocab(
    "tens", "twentieth thirtieth fortieth fiftieth sixtieth seventieth eightieth ninetieth", _TENS
)
_vocab("tens", "twenties thirties forties fifties sixties seventies eighties nineties", _TENS)
_vocab("hundred", "hundred hundreds hundredth", [100, 100, 100])
_vocab("scale", "dozen dozens", [12, 12])
for _i, _name in enumerate(("thousand", "million", "billion", "trillion"), start=1):
    _vocab("scale", f"{_name} {_name}s {_name}th", [1000**_i] * 3)
_vocab("frac", "half halves quarter quarters", [Fraction(1, 2)] * 2 + [Fraction(1, 4)] * 2)

_MULTIPLIERS = frozenset(w for w, (k, _) in _NUMBER_KINDS.items() if k in ("hundred", "scale"))
_FRACS = frozenset(w for w, (k, _) in _NUMBER_KINDS.items() if k == "frac")
_DIGITS = frozenset(w for w, (k, v) in _NUMBER_KINDS.items() if k == "unit") | {"oh"}
# num2words writes "one thousand, two hundred": a comma right after these is not a boundary
_SCALE_WORDS = frozenset(w for w, (k, _) in _NUMBER_KINDS.items() if k == "scale")
# Spoken alone these are often heard as something else; forgiven only for that something else.
_HOMOPHONES = {
    "one": frozenset({"won", "wan", "a", "an", "on"}),
    "two": frozenset({"to", "too", "tu"}),
    "four": frozenset({"for", "fore", "far"}),
    "eight": frozenset({"ate", "ait"}),
}

Values = tuple[Fraction, ...]
Span = tuple[int, int]


@dataclass(frozen=True)
class Verdict:
    ok: bool
    similarity: float  # 0..1 character similarity after numbers are reconciled
    reason: str | None  # plain words for a failure; None when ok


# -- folding -----------------------------------------------------------------------------------


def _tokens(text: str) -> tuple[list[str], list[bool]]:
    """Spoken words + "a boundary (punctuation) follows this word" flags."""
    spoken = normalize_text(_GLUED_UNIT.sub(r"\1 \2", text)).casefold()
    while (unstuttered := _STUTTER.sub("", spoken)) != spoken:
        spoken = unstuttered
    for pattern, repl in _SPOKEN_FORMS:
        spoken = pattern.sub(repl, spoken)
    words: list[str] = []
    bounds: list[bool] = []
    for tok in _TOKEN.findall(spoken):
        if tok in _BOUNDARY:
            if words and not (tok == "," and words[-1] in _SCALE_WORDS):
                bounds[-1] = True
        else:
            words.append(tok)
            bounds.append(False)
    return words, bounds


def fold(text: str) -> str:
    """Spoken-word form: normalizer rules, lowercase, contractions opened, no punctuation."""
    return " ".join(_tokens(text)[0])


def similarity(expected: str, transcript: str) -> float:
    """Character similarity in [0, 1] of the folded strings (no number reconciliation)."""
    return Indel.normalized_similarity(fold(expected), fold(transcript))


# -- numbers -----------------------------------------------------------------------------------


def _spans(words: list[str], bounds: list[bool]) -> list[Span]:
    """[start, end) word ranges of number phrases, never crossing punctuation."""
    spans: list[Span] = []
    i, n = 0, len(words)

    def joined(j: int, count: int) -> bool:  # words[j:j+count] all inside one clause
        return j + count <= n and not any(bounds[j : j + count - 1])

    while i < n:
        word = words[i]
        starts = word in _NUMBER_KINDS or (
            word in ("a", "an")
            and joined(i, 2)
            and (words[i + 1] in _MULTIPLIERS or words[i + 1] in _FRACS)
        )
        if not starts:
            i += 1
            continue
        j = i + 1
        while j < n and not bounds[j - 1]:
            w = words[j]
            if w in _NUMBER_KINDS:
                j += 1
            elif w in ("and", "oh") and joined(j, 2) and words[j + 1] in _NUMBER_KINDS:
                j += 1
            elif (
                w == "point"
                and joined(j, 2)
                and words[j + 1] in _DIGITS
                and not (j == i + 1 and word == "one" and i > 0 and words[i - 1] == "at")
            ):
                j += 1
            elif (
                w == "and"
                and joined(j, 3)
                and words[j + 1] in ("a", "an")
                and (words[j + 2] in _FRACS)
            ):
                j += 3  # "two and a half"
            else:
                break
        spans.append((i, j))
        i = j
    return spans


def _greedy(words: list[str]) -> list[tuple[Fraction, bool]]:
    """Read number words (no "point") left to right into one or more numbers. Each comes with a
    flag: True when it is a plain 0-99 number (no hundred/scale), so two may form a year."""
    out: list[tuple[Fraction, bool]] = []
    total = cur = Fraction(0)
    last: str | None = None
    plain = True
    add_fraction = False

    def flush() -> None:
        nonlocal total, cur, last, plain
        if last is not None:
            out.append((total + cur, plain and total == 0 and cur < 100))
        total = cur = Fraction(0)
        last, plain = None, True

    for k, word in enumerate(words):
        if word == "and":
            continue
        if word == "oh":
            flush()
            continue
        if word in ("a", "an"):
            if last is not None and k + 1 < len(words) and words[k + 1] in _FRACS:
                add_fraction = True  # "two and a half"
                continue
            flush()
            cur, last = Fraction(1), "unit"
            continue
        kind, value = _NUMBER_KINDS[word]
        if kind == "frac":
            if add_fraction:
                cur += value
                add_fraction, plain = False, False
            elif last in ("unit", "teen") and total == 0:
                cur *= value  # "three quarters", "a half"
                plain = False
            else:
                flush()
                out.append((value, False))
            continue
        if kind == "unit" and last in (None, "tens", "hundred", "scale"):
            cur += value
        elif kind in ("teen", "tens") and last in (None, "hundred", "scale"):
            cur += value
        elif kind == "hundred" and last in (None, "unit", "teen", "tens"):
            cur = (cur or 1) * value
            plain = False
        elif kind == "scale" and last is not None and last != "scale":
            total += (cur or 1) * value
            cur = Fraction(0)
            plain = False
        elif kind == "scale" and last is None:
            total, plain = value, False
        else:
            flush()
            if kind in ("hundred", "scale"):
                cur, plain = value, False
            else:
                cur = value
        last = kind
    flush()
    return out


def _plain_readings(words: list[str]) -> set[Values]:
    """Readings of a number phrase without "point"."""
    nums = _greedy(words)
    readings = {tuple(v for v, _ in nums)}
    # years: "nineteen eighty-four" -> 1984, "nineteen oh five" -> 1905, "twenty twenty" -> 2020
    merged: list[Fraction] = []
    i = 0
    while i < len(nums):
        if (
            i + 1 < len(nums)
            and nums[i][1]
            and nums[i + 1][1]
            and 10 <= nums[i][0] <= 99
            and (nums[i + 1][0] >= 10 or "oh" in words)
        ):
            merged.append(nums[i][0] * 100 + nums[i + 1][0])
            i += 2
        else:
            merged.append(nums[i][0])
            i += 1
    readings.add(tuple(merged))
    # digit strings: "nine one one" -> 911
    digit_words = [w for w in words if w != "and"]
    if len(digit_words) >= 2 and all(
        w == "oh" or _NUMBER_KINDS.get(w, ("", 0))[0] == "unit" for w in digit_words
    ):
        digits = "".join("0" if w == "oh" else str(int(_NUMBER_KINDS[w][1])) for w in digit_words)
        readings.add((Fraction(int(digits)),))
    # a written fraction "1/2" folds to "one two"
    if len(nums) == 2 and 1 <= nums[0][0] < nums[1][0] and nums[1][0] in (2, 3, 4, 8):
        readings.add((nums[0][0] / nums[1][0],))
    return readings


def _readings(words: list[str]) -> set[Values]:
    """Every value tuple a number phrase can plausibly mean."""
    if "point" not in words:
        return _plain_readings(words)
    k = words.index("point")
    head_words, after = words[:k], words[k + 1 :]
    n_digits = 0
    while n_digits < len(after) and after[n_digits] in _DIGITS:
        n_digits += 1
    digits = "".join("0" if w == "oh" else str(int(_NUMBER_KINDS[w][1])) for w in after[:n_digits])
    rest = after[n_digits:]
    head = [v for v, _ in _greedy(head_words)]
    if not head or not digits:  # "point" with nothing readable around it: read the halves
        left = _plain_readings(head_words) if head_words else {()}
        right = _readings(after) if after else {()}
        return {a + b for a in left for b in right}
    value = head[-1] + Fraction(int(digits), 10 ** len(digits))
    if rest and all(w in _MULTIPLIERS for w in rest):  # "two point five million"
        for w in rest:
            value *= _NUMBER_KINDS[w][1]
        rest = []
    readings = {(*head[:-1], value)}
    if not rest and len(digits) == 2 and 1 <= head[-1] <= 12:
        readings.add((*head, Fraction(int(digits))))  # a time: "7.15" for "seven fifteen"
    if rest:
        tails = _readings(rest)
        readings = {r + t for r in readings for t in tails}
    return readings


def _align_numbers(
    said: list[set[Values]],
    heard: list[set[Values]],
    s_adjacent: list[bool],
    h_adjacent: list[bool],
) -> tuple[list[tuple[range, range]], list[tuple[list[int], list[int]]]]:
    """Order-preserving best matching of number spans (most spans explained by equal values).

    Up to MAX_GROUP consecutive spans on one side may match one span on the other: the
    normalizer drops the commas in "1912, 1500" and "4, 8, 15", so whisper's digits can read as
    one phrase where the source had several. Returns the matched groups and, around them, the
    unmatched said/heard span indices. ``*_adjacent[k]`` says span k+1 directly follows span k
    (only punctuation or "and" between): only such runs of phrases may group."""
    n, m = len(said), len(heard)
    joined: dict[tuple[int, int, int], set[Values]] = {}

    def readings(side: int, a: int, b: int) -> set[Values]:
        key = (side, a, b)
        if key not in joined:
            spans, adjacent = (said, s_adjacent) if side == 0 else (heard, h_adjacent)
            out: set[Values] = {()}
            for k in range(a, b):
                if k > a and not adjacent[k - 1] or len(out) > MAX_READINGS:
                    out = set()  # not a list, or too ambiguous to enumerate: no group
                    break
                out = {head + tail for head in out for tail in spans[k]}
            joined[key] = out
        return joined[key]

    def longest(side: int, k: int) -> int:
        return max(len(r) for r in (said if side == 0 else heard)[k])

    best = [[0] * (m + 1) for _ in range(n + 1)]
    move = [[(0, 0, False)] * (m + 1) for _ in range(n + 1)]
    for i in range(n, -1, -1):
        for j in range(m, -1, -1):
            if i == n and j == m:
                continue
            options: list[tuple[int, bool, tuple[int, int, bool]]] = []
            if i < n:
                options.append((best[i + 1][j], False, (1, 0, False)))
            if j < m:
                options.append((best[i][j + 1], False, (0, 1, False)))
            groups = [(1, 1)]
            if j < m:
                groups += [(d, 1) for d in range(2, MAX_GROUP + 1) if longest(1, j) >= d]
            if i < n:
                groups += [(1, d) for d in range(2, MAX_GROUP + 1) if longest(0, i) >= d]
            for di, dj in groups:
                if i + di > n or j + dj > m:
                    continue
                if readings(0, i, i + di) & readings(1, j, j + dj):
                    options.append((best[i + di][j + dj] + di + dj, True, (di, dj, True)))
            best[i][j], _, move[i][j] = max(options, key=lambda o: (o[0], o[1]))
    matched: list[tuple[range, range]] = []
    gaps: list[tuple[list[int], list[int]]] = []
    lone_s: list[int] = []
    lone_h: list[int] = []
    i = j = 0
    while i < n or j < m:
        di, dj, is_match = move[i][j]
        if is_match:
            gaps.append((lone_s, lone_h))
            lone_s, lone_h = [], []
            matched.append((range(i, i + di), range(j, j + dj)))
        else:
            lone_s.extend(range(i, i + di))
            lone_h.extend(range(j, j + dj))
        i, j = i + di, j + dj
    gaps.append((lone_s, lone_h))
    return matched, gaps


def _adjacency(words: list[str], spans: list[Span]) -> list[bool]:
    """[k] -> span k+1 follows span k with nothing but "and" between (a dropped-comma list)."""
    return [
        all(w == "and" for w in words[spans[k][1] : spans[k + 1][0]]) for k in range(len(spans) - 1)
    ]


def _is_homophone_number(words: list[str], span: Span) -> bool:
    return span[1] - span[0] == 1 and words[span[0]] in _HOMOPHONES


def _number_problem(
    s_words: list[str],
    h_words: list[str],
    s_spans: list[Span],
    h_spans: list[Span],
    gaps: list[tuple[list[int], list[int]]],
) -> str | None:
    """Unmatched number phrases. A lone homophone-number ("two") is left to the word check,
    which forgives it only when its homophone ("to") stands in its place."""

    def text(words: list[str], span: Span) -> str:
        return " ".join(words[span[0] : span[1]])

    for lone_s, lone_h in gaps:
        if lone_s and lone_h:
            said, heard = text(s_words, s_spans[lone_s[0]]), text(h_words, h_spans[lone_h[0]])
            return f"wrong number: {said!r} heard as {heard!r}"
        for k in lone_s:
            if not _is_homophone_number(s_words, s_spans[k]):
                return f"number missing: {text(s_words, s_spans[k])!r} not heard"
        for k in lone_h:
            if not _is_homophone_number(h_words, h_spans[k]):
                return f"extra number: {text(h_words, h_spans[k])!r} heard"
    return None


def _group_tags(groups: list[tuple[range, range]], *, side: int) -> dict[int, str]:
    """span index -> placeholder token; a group's first span carries it, the rest are dropped."""
    tags: dict[int, str] = {}
    for n, group in enumerate(groups):
        for pos, k in enumerate(group[side]):
            tags[k] = f"\x00{n}" if pos == 0 else ""
    return tags


def _collapse(words: list[str], spans: list[Span], tags: dict[int, str]) -> list[str]:
    """Replace each matched number span with its group's shared placeholder token."""
    out: list[str] = []
    cursor = 0
    for k, (start, end) in enumerate(spans):
        out.extend(words[cursor:start])
        if k not in tags:
            out.extend(words[start:end])
        elif tags[k]:
            out.append(tags[k])
        cursor = end
    out.extend(words[cursor:])
    return out


# -- words -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Run:
    said: list[str]
    heard: list[str]
    s_at: int  # said index where the run starts
    h_at: int


def _runs(said: list[str], heard: list[str]) -> list[_Run]:
    """Maximal stretches of non-matching words under an optimal word alignment."""
    runs: list[_Run] = []
    current: _Run | None = None
    for op in Levenshtein.opcodes(said, heard):
        if op.tag == "equal":
            current = None
            continue
        if current is None:
            current = _Run([], [], op.src_start, op.dest_start)
            runs.append(current)
        current.said.extend(said[op.src_start : op.src_end])
        current.heard.extend(heard[op.dest_start : op.dest_end])
    return runs


def _neighbours(words: list[str], at: int, length: int) -> set[str]:
    return {words[k] for k in (at - 1, at + length) if 0 <= k < len(words)}


def _negation_problem(run: _Run, said: list[str], heard: list[str]) -> str | None:
    lost = sorted((set(run.said) & _NEGATIONS) - set(run.heard))
    added = sorted((set(run.heard) & _NEGATIONS) - set(run.said))
    # a repeat collapsed or doubled next to the same word ("no, no, no" heard as "no, no")
    lost = [w for w in lost if w not in _neighbours(said, run.s_at, len(run.said))]
    added = [w for w in added if w not in _neighbours(heard, run.h_at, len(run.heard))]
    if lost:
        return f"negation missing: {lost[0]!r} not heard"
    if added:
        return f"negation added: {added[0]!r} heard"
    if len(run.said) == 1 and len(run.heard) == 1:
        a, b = run.said[0], run.heard[0]
        for long, short in ((a, b), (b, a)):
            if len(short) >= 4 and any(long == p + short for p in _NEGATING_PREFIXES):
                return f"meaning flipped: {a!r} heard as {b!r}"
    return None


def _homophone_problem(run: _Run) -> str | None:
    """A lone homophone-number is forgiven only when its homophone stands in its place."""
    for words, other, verb in ((run.said, run.heard, "not heard"), (run.heard, run.said, "heard")):
        for w in words:
            if w in _HOMOPHONES and not any(
                o in _HOMOPHONES[w] or (o != w and o.startswith(w)) for o in other
            ):
                label = "number missing" if verb == "not heard" else "extra number"
                return f"{label}: {w!r} {verb}"
    return None


def _skeleton(word: str) -> str:
    """Consonants only, doubled letters collapsed, s ~ z: colour/color, realised/realized."""
    out = re.sub(r"[aeiouy]", "", word.replace("z", "s"))
    return re.sub(r"(.)\1+", r"\1", out)


def _cost(run: _Run) -> tuple[int, bool]:
    """(errors, counts toward the word-gap rule) for one run."""
    if len(run.said) == 1 and len(run.heard) == 1:
        a, b = run.said[0], run.heard[0]
        if min(len(a), len(b)) >= 4 and Indel.normalized_similarity(a, b) >= SAME_WORD_SIMILARITY:
            if a[0] == b[0] and _skeleton(a) == _skeleton(b):
                return 0, False  # same word, other spelling: colour / color
            return 1, False  # a near-miss word (east / west): an error, but not a gap
    if (
        run.said
        and run.heard
        and max(len(run.said), len(run.heard)) <= SPELLING_VARIANT_MAX_WORDS
        and Indel.normalized_similarity("".join(run.said), "".join(run.heard))
        >= SPELLING_VARIANT_SIMILARITY
    ):
        return 1, False  # a respelled or split name
    return max(len(run.said), len(run.heard)), True


def _loop(run: _Run, heard: list[str]) -> str | None:
    """Extra words that exactly repeat the words next to them: an engine looping."""
    size = len(run.heard)
    if run.said or size < 2:
        return None
    before = heard[run.h_at - size : run.h_at]
    after = heard[run.h_at + size : run.h_at + 2 * size]
    if run.heard in (before, after):
        return f"repeated words: {' '.join(run.heard)!r}"
    return None


def _edge_cut(run: _Run, said: list[str]) -> str | None:
    """Words missing at the very start or end: an engine that started late or stopped early."""
    if run.heard or not run.said:
        return None
    at_start, at_end = run.s_at == 0, run.s_at + len(run.said) == len(said)
    if not (at_start or at_end):
        return None
    lost = [w for w in run.said if w not in _FILLERS and not _VOCAL_SOUND.fullmatch(w)]
    if len(lost) >= 2 or (lost and len(lost[0]) > 3):
        where = "end" if at_end else "start"
        return f"the {where} was cut off: {' '.join(run.said)!r} not heard"
    return None


def _score_one(expected: str, transcript: str, *, min_ratio: float, max_word_gap: int) -> Verdict:
    s_words, s_bounds = _tokens(expected)
    h_words, h_bounds = _tokens(transcript)
    s_spans, h_spans = _spans(s_words, s_bounds), _spans(h_words, h_bounds)
    s_read = [_readings(s_words[a:b]) for a, b in s_spans]
    h_read = [_readings(h_words[a:b]) for a, b in h_spans]
    groups, gaps = _align_numbers(
        s_read, h_read, _adjacency(s_words, s_spans), _adjacency(h_words, h_spans)
    )
    problem = _number_problem(s_words, h_words, s_spans, h_spans, gaps)

    said = _collapse(s_words, s_spans, _group_tags(groups, side=0))
    heard = _collapse(h_words, h_spans, _group_tags(groups, side=1))
    sim = Indel.normalized_similarity(" ".join(said), " ".join(heard))

    worst_gap = errors = 0
    for run in _runs(said, heard):
        if "".join(run.said) == "".join(run.heard):
            continue  # only spacing differs: "never mind" / "nevermind", "no one" / "noone"
        problem = (
            problem
            or _negation_problem(run, said, heard)
            or _homophone_problem(run)
            or _edge_cut(run, said)
            or _loop(run, heard)
        )
        cost, is_gap = _cost(run)
        errors += cost
        if is_gap:
            worst_gap = max(worst_gap, cost)
    budget = max(MIN_ERROR_BUDGET, math.ceil(ERROR_BUDGET * len(said)))
    if problem is None and worst_gap > max_word_gap:
        problem = f"{worst_gap} words in a row differ from the text"
    if problem is None and errors > budget:
        problem = f"{errors} of {len(said)} words differ from the text"
    if problem is None and sim < min_ratio:
        problem = f"similarity {sim:.2f} below {min_ratio:.2f}"
    return Verdict(ok=problem is None, similarity=sim, reason=problem)


def score(
    expected: str,
    transcript: str,
    *,
    alternates: Sequence[str] = (),
    min_ratio: float = 0.85,
    max_word_gap: int = DEFAULT_MAX_WORD_GAP,
) -> Verdict:
    """Judge ``transcript`` against ``expected`` (and any ``alternates`` — e.g. the text without
    pronunciation respellings); the best verdict wins: a pass over a fail, then similarity."""
    verdicts = [
        _score_one(text, transcript, min_ratio=min_ratio, max_word_gap=max_word_gap)
        for text in dict.fromkeys((expected, *alternates))
    ]
    return max(verdicts, key=lambda v: (v.ok, v.similarity))
