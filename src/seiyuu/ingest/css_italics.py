"""Dependency-light CSS italic resolver (THOUGHT Phase 2a).

Phase 1 captured only inline ``<em>``/``<i>``. Many EPUBs instead declare interior
monologue with a CSS class (``.italic``, ``.calibre3 { font-style: oblique }``) or an
inline ``style="font-style:italic"``. This module builds a book-global set of class names
that resolve to italic, so the ingest italic walker can widen its per-character italic flag
to those CSS/inline signals — feeding the SAME ``Block.italic_spans`` the downstream
thought pipeline already consumes, with no prompt/schema/cache change.

Conservative by construction — the ONLY error direction is "miss an italic", never
"invent one":
  - Only class selectors count toward the class set; bare element (``p``, ``div``) and
    universal (``*``) selectors are dropped, so a whole-paragraph italic (emphasis of
    everything, or a letter/telegraph) never becomes a mid-prose thought candidate.
  - Any class ever forced ``font-style: normal`` ANYWHERE is subtracted from the italic
    set (no specificity math — exclude the ambiguous).
  - ``@media``/``@font-face``/``@keyframes``/``@supports`` bodies are skipped whole, and
    the ``font:`` shorthand is not parsed — both are deliberate, precision-safe misses.

No new dependency: a small regex parser (no cssutils/tinycss2). Kept out of ``epub.py`` so
the parser is unit-testable in isolation.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass

# Property/value keywords are matched case-INSENSITIVELY (CSS is ASCII-case-insensitive for
# these); class TOKENS are kept VERBATIM because HTML class selectors are case-sensitive.
_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_FONT_STYLE_DECL = re.compile(r"font-style\s*:\s*([^;{}]+)", re.IGNORECASE)
_CLASS_TOKEN = re.compile(r"\.([A-Za-z0-9_-]+)")
# An ``em``/``i`` element selector (bare or in a descendant/combinator chain), not a class
# or id fragment. Used ONLY to record the informational forced-normal flags.
_EM_I_ELEMENT = re.compile(r"(?:^|[\s>+~,])(em|i)(?=$|[\s>+~,.:#\[])", re.IGNORECASE)
# CSS combinators separating compound selectors: descendant (whitespace), child (``>``),
# and the adjacent/general siblings (``+``/``~``). The compound to the RIGHT of the last
# combinator is the SUBJECT (the matched element); everything left is an ancestor/sibling
# CONDITION and must not be harvested as if it were the styled element.
_COMBINATOR = re.compile(r"[\s>+~]+")
# A generated-content pseudo-element (``::before``/``::after`` and their legacy single-colon
# spellings). ``font-style`` on these styles GENERATED content only, never the element's own
# characters — so a class carrying such a rule must not be harvested as italic.
_PSEUDO_ELEMENT = re.compile(r"::?(?:before|after)\b", re.IGNORECASE)


def _subject_compound(selector: str) -> str:
    """Reduce a complex selector to its rightmost (SUBJECT) compound.

    ``.dialogue .whisper`` matches a ``.whisper`` INSIDE a ``.dialogue``; only ``.whisper``
    is the styled element, so ``.dialogue`` must not enter the class set. Splitting on
    combinators and taking the last part scopes harvesting to the subject alone.
    """
    parts = _COMBINATOR.split(selector.strip())
    return parts[-1] if parts else ""


# Inline ``style="..."`` probes for the per-element font-style declaration.
INLINE_ITALIC = re.compile(r"font-style\s*:\s*(?:italic|oblique)", re.IGNORECASE)
INLINE_NORMAL = re.compile(r"font-style\s*:\s*normal", re.IGNORECASE)


@dataclass(frozen=True)
class ItalicStyleMap:
    """Book-global italic signal harvested from CSS (external sheets + ``<style>`` blocks).

    ``italic_classes`` holds every class token that resolves to italic and is NEVER forced
    normal anywhere. ``em_forced_normal``/``i_forced_normal`` are informational only: they
    record that a sheet restyled ``<em>``/``<i>`` to normal, but by design we KEEP the tag
    italic signal (byte-identical to Phase 1) and do not act on these flags.
    """

    italic_classes: frozenset[str] = frozenset()
    em_forced_normal: bool = False
    i_forced_normal: bool = False


def _classify_font_style(body: str) -> str | None:
    """LAST ``font-style`` in a declaration block wins (in-block cascade).

    Returns ``"italic"`` (italic/oblique), ``"normal"`` (normal-forcing), or ``None`` when
    the block declares no recognizable ``font-style`` (e.g. only the ``font:`` shorthand).
    """
    matches = _FONT_STYLE_DECL.findall(body)
    if not matches:
        return None
    tokens = matches[-1].strip().lower().split()
    keyword = tokens[0] if tokens else ""
    if keyword in ("italic", "oblique"):
        return "italic"
    if keyword == "normal":
        return "normal"
    return None


def _iter_top_level_rules(css: str) -> Iterable[tuple[str, str]]:
    """Yield ``(prelude, body)`` for each TOP-LEVEL ``selectors { decls }`` rule.

    Brace-DEPTH-aware (never ``split('}')``): the entire balanced body of any ``@``-rule
    (``@media``/``@font-face``/``@keyframes``/``@supports``) is consumed and SKIPPED, so no
    rule nested inside an at-rule is ever harvested.
    """
    i = 0
    n = len(css)
    prelude_start = 0
    while i < n:
        c = css[i]
        if c == "{":
            prelude = css[prelude_start:i].strip()
            depth = 1
            j = i + 1
            while j < n and depth > 0:
                if css[j] == "{":
                    depth += 1
                elif css[j] == "}":
                    depth -= 1
                j += 1
            body = css[i + 1 : j - 1] if depth == 0 else css[i + 1 : j]
            if not prelude.startswith("@"):
                yield prelude, body
            i = j
            prelude_start = i
        elif c == "}":
            # Stray close brace — reset the prelude accumulator.
            i += 1
            prelude_start = i
        else:
            i += 1


def parse_css_italics(sheets: Iterable[str]) -> ItalicStyleMap:
    """Union CSS blobs into one :class:`ItalicStyleMap`.

    ``sheets`` are raw CSS strings (external stylesheet contents and/or ``<style>`` block
    text). Classes forced normal in ANY sheet are subtracted from the italic set collected
    across ALL sheets, so cross-sheet resets win (precision over recall).
    """
    italic: set[str] = set()
    normal: set[str] = set()
    em_forced_normal = False
    i_forced_normal = False

    for raw in sheets:
        css = _COMMENT.sub("", raw)
        for prelude, body in _iter_top_level_rules(css):
            classification = _classify_font_style(body)
            if classification is None:
                continue
            for selector in prelude.split(","):
                subject = _subject_compound(selector)
                # A ``font-style`` on a ``::before``/``::after`` styles generated content,
                # not the matched element's own text — never harvest such a class.
                if _PSEUDO_ELEMENT.search(subject) is not None:
                    continue
                tokens = _CLASS_TOKEN.findall(subject)
                if classification == "italic":
                    italic.update(tokens)
                else:  # normal-forcing
                    normal.update(tokens)
                    m = _EM_I_ELEMENT.search(subject)
                    if m is not None:
                        if m.group(1).lower() == "em":
                            em_forced_normal = True
                        else:
                            i_forced_normal = True

    return ItalicStyleMap(
        italic_classes=frozenset(italic - normal),
        em_forced_normal=em_forced_normal,
        i_forced_normal=i_forced_normal,
    )


# A book with no CSS italics reduces the ingest walker to Phase-1 tag-only behavior.
EMPTY_ITALIC_MAP = ItalicStyleMap()
