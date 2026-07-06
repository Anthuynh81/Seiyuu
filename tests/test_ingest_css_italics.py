"""Ingest-level integration + adversarial precision suite for THOUGHT Phase 2a.

Phase 2a widens the ingest italic signal from inline ``<em>``/``<i>`` to CSS-class and
inline-``style`` italics, feeding the SAME ``Block.italic_spans`` Phase 1 already produces.
The contract each positive case pins: a CSS/inline italic yields EXACTLY the italic_spans an
equivalent ``<em>`` wrapping would (offset geometry is provenance-agnostic). The adversarial
cases pin the precision guardrails — bare/universal selectors, @-rule bodies, whole-element
italic, normal resets, and decomposed captions produce NO span (under-capture, never over).
"""

from pathlib import Path

from ebooklib import epub

from seiyuu.attribute.spans import thought_candidate_spans
from seiyuu.attribute.validate import reconstructs_block
from seiyuu.ingest.css_italics import EMPTY_ITALIC_MAP, parse_css_italics
from seiyuu.ingest.epub import _build_italic_style_map, _extract_doc_blocks, parse_epub

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _blocks(body: str, italic_map=EMPTY_ITALIC_MAP):
    return _extract_doc_blocks(f"<html><body>{body}</body></html>".encode(), italic_map)


def _map(*sheets: str):
    return parse_css_italics(sheets)


def _assert_matches_em(css_body: str, em_body: str, italic_map):
    """The CSS/inline form yields identical text AND italic_spans to the <em> equivalent."""
    (css_block,) = _blocks(css_body, italic_map)
    (em_block,) = _blocks(em_body, EMPTY_ITALIC_MAP)
    assert css_block.text == em_block.text
    assert css_block.italic_spans == em_block.italic_spans
    return css_block


# --------------------------------------------------------------------------------------
# Positive: every CSS/inline italic form matches the <em> equivalent exactly.
# --------------------------------------------------------------------------------------


def test_class_italic_matches_em_wrapping():
    m = _map(".italic { font-style: italic }")
    block = _assert_matches_em(
        '<p>She paused. <span class="italic">I must leave now.</span></p>',
        "<p>She paused. <em>I must leave now.</em></p>",
        m,
    )
    ((s, e),) = block.italic_spans
    assert block.text[s:e] == "I must leave now."


def test_oblique_class_matches_em_wrapping():
    m = _map(".calibre3 { font-style: oblique }")
    _assert_matches_em(
        '<p>He read <span class="calibre3">The manuscript</span> again.</p>',
        "<p>He read <em>The manuscript</em> again.</p>",
        m,
    )


def test_inline_style_italic_on_inner_span_matches_em():
    _assert_matches_em(
        '<p>She paused. <span style="font-style:italic">I must leave now.</span></p>',
        "<p>She paused. <em>I must leave now.</em></p>",
        EMPTY_ITALIC_MAP,
    )


def test_inline_oblique_matches_em():
    _assert_matches_em(
        '<p>Then <span style="font-style: oblique">a quiet thought</span> came.</p>',
        "<p>Then <em>a quiet thought</em> came.</p>",
        EMPTY_ITALIC_MAP,
    )


def test_multi_class_only_the_italic_one_counts():
    m = _map(".b { font-style: italic }")
    _assert_matches_em(
        '<p>x <span class="a b c">the middle word</span> y.</p>',
        "<p>x <em>the middle word</em> y.</p>",
        m,
    )


def test_nested_em_inside_italic_class_is_one_merged_run():
    m = _map(".thought { font-style: italic }")
    block = _assert_matches_em(
        '<p>She knew <span class="thought">I must <em>really</em> leave</span> now.</p>',
        "<p>She knew <em>I must really leave</em> now.</p>",
        m,
    )
    assert len(block.italic_spans) == 1  # nested tag inside an italic class merges, not splits


# --------------------------------------------------------------------------------------
# Inheritance: nearest-ancestor-wins, and a nearer `normal` cancels an ancestor italic.
# --------------------------------------------------------------------------------------


def test_nearer_inline_normal_cancels_ancestor_italic_class():
    m = _map(".thought { font-style: italic }")
    _assert_matches_em(
        '<p>She thinks <span class="thought">I must '
        '<span style="font-style:normal">not</span> leave</span> now.</p>',
        "<p>She thinks <em>I must </em><span>not</span><em> leave</em> now.</p>",
        m,
    )


def test_italic_class_inherits_to_descendants():
    # font-style inherits: a class on a wrapper italicizes a plain inner span.
    m = _map(".thought { font-style: italic }")
    _assert_matches_em(
        '<p><span class="thought">I must <span>truly</span> leave</span></p>',
        "<p><em>I must truly leave</em></p>",
        m,
    )


# --------------------------------------------------------------------------------------
# Adversarial precision: only under-capture is allowed. Each yields NO italic span.
# --------------------------------------------------------------------------------------


def test_class_reset_normal_in_second_sheet_yields_no_span():
    m = _map(".x { font-style: italic }", ".x { font-style: normal }")
    (block,) = _blocks('<p>text <span class="x">word</span> more.</p>', m)
    assert block.italic_spans == []


def test_bare_element_italic_selector_is_ignored():
    m = _map("p { font-style: italic }")
    assert m.italic_classes == frozenset()
    (block,) = _blocks("<p>A whole paragraph the sheet italicizes at the element level.</p>", m)
    assert block.italic_spans == []


def test_at_media_scoped_italic_class_is_ignored():
    m = _map("@media screen { .thought { font-style: italic } }")
    assert m.italic_classes == frozenset()
    (block,) = _blocks('<p>x <span class="thought">word</span> y.</p>', m)
    assert block.italic_spans == []


def test_whole_paragraph_inline_italic_is_invisible():
    # style on the <p> itself: the ancestor walk stops at el, so a letter/telegraph
    # paragraph never becomes a mid-prose thought candidate.
    (block,) = _blocks('<p style="font-style:italic">A whole italic paragraph here.</p>')
    assert block.text == "A whole italic paragraph here."
    assert block.italic_spans == []


def test_whole_paragraph_italic_class_is_invisible():
    m = _map(".letter { font-style: italic }")
    (block,) = _blocks('<p class="letter">A whole italic paragraph here.</p>', m)
    assert block.italic_spans == []


def test_italic_class_on_caption_is_decomposed_before_the_walk():
    # SKIP_CLASS_PATTERN/nav decompose runs BEFORE the italic walk: caption text (and its
    # italic) never reaches a block.
    m = _map(".thought { font-style: italic }")
    blocks = _blocks(
        '<div class="caption"><p class="thought">HIDDEN CAPTION</p></div>'
        "<p>Visible prose here.</p>",
        m,
    )
    assert len(blocks) == 1
    assert blocks[0].text == "Visible prose here."
    assert blocks[0].italic_spans == []


# --------------------------------------------------------------------------------------
# Byte-identity: an EMPTY ItalicStyleMap is exactly Phase-1 tag-only behavior.
# --------------------------------------------------------------------------------------


def test_empty_map_reproduces_tag_only_spans_and_ignores_classes():
    body = '<p>She paused. <em>I must leave now.</em> <span class="italic">not italic</span></p>'
    (block,) = _blocks(body, EMPTY_ITALIC_MAP)
    # The <em> is still captured; the class span is invisible under an empty map.
    ((s, e),) = block.italic_spans
    assert block.text[s:e] == "I must leave now."


def test_empty_map_matches_default_arg():
    body = "<p>Plain <em>emphasis</em> here.</p>"
    (a,) = _blocks(body)  # default EMPTY_ITALIC_MAP
    (b,) = _blocks(body, EMPTY_ITALIC_MAP)
    assert a.text == b.text
    assert a.italic_spans == b.italic_spans


# --------------------------------------------------------------------------------------
# Downstream candidate + reconstruction path is reused unchanged for a CSS-italic block.
# --------------------------------------------------------------------------------------


def test_css_italic_block_feeds_candidate_and_reconstructs():
    m = _map(".thought { font-style: italic }")
    text = "She froze. I have to get out of here. Her heart raced."
    body = (
        '<p>She froze. <span class="thought">I have to get out of here.</span> Her heart raced.</p>'
    )
    (block,) = _blocks(body, m)
    assert block.text == text
    parts = thought_candidate_spans("ch001_b0001", block.text, block.italic_spans)
    # The substantial CSS-italic sentence becomes a THOUGHT candidate, exactly as an <em> would.
    cands = [p for p in parts if p.candidate_id]
    assert len(cands) == 1
    assert cands[0].text == "I have to get out of here."
    # Reconstruction still holds: the partition reproduces the block text.
    assert reconstructs_block(block.text, [p.text for p in parts])


# --------------------------------------------------------------------------------------
# Pre-pass: external ITEM_STYLE sheet + in-document <style> unioned once per book.
# --------------------------------------------------------------------------------------


def _build_epub(path: Path, *, css_content: str | None, doc_html: str) -> Path:
    book = epub.EpubBook()
    book.set_identifier("css-italics-test")
    book.set_title("CSS Italics Test")
    book.set_language("en")
    book.add_author("Test Author")
    if css_content is not None:
        book.add_item(
            epub.EpubItem(
                uid="style",
                file_name="style.css",
                media_type="text/css",
                content=css_content.encode(),
            )
        )
    doc = epub.EpubHtml(title="Chapter 1", file_name="ch1.xhtml", lang="en")
    doc.set_content(doc_html)
    book.add_item(doc)
    book.spine = [doc]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    epub.write_epub(str(path), book)
    return path


def test_external_stylesheet_class_captured_by_prepass(tmp_path):
    doc_html = (
        "<html><body><h2>Chapter 1</h2>"
        '<p>She froze. <span class="thought">I have to get out of here.</span> '
        "Her heart raced and the room felt smaller with every passing second here.</p>"
        "</body></html>"
    )
    path = _build_epub(
        tmp_path / "ext.epub",
        css_content=".thought { font-style: italic }",
        doc_html=doc_html,
    )
    result = parse_epub(path)
    block = next(
        b for ch in result.book.chapters for b in ch.blocks if "I have to get out" in b.text
    )
    ((s, e),) = block.italic_spans
    assert block.text[s:e] == "I have to get out of here."


def test_in_document_style_block_captured_by_prepass(tmp_path):
    # Note: <style> is placed in <body> — ebooklib's WRITER regenerates <head> and drops a
    # head <style>; a real EPUB preserves it in either place and the pre-pass reads both.
    doc_html = (
        "<html><body><style>.aside { font-style: italic }</style>"
        "<h2>Chapter 1</h2>"
        '<p>She froze. <span class="aside">I have to get out of here.</span> '
        "Her heart raced and the room felt smaller with every passing second here.</p>"
        "</body></html>"
    )
    path = _build_epub(tmp_path / "instyle.epub", css_content=None, doc_html=doc_html)
    result = parse_epub(path)
    block = next(
        b for ch in result.book.chapters for b in ch.blocks if "I have to get out" in b.text
    )
    ((s, e),) = block.italic_spans
    assert block.text[s:e] == "I have to get out of here."
    # And the CSS text itself never leaks into narration.
    assert all("font-style" not in b.text for ch in result.book.chapters for b in ch.blocks)


def test_prepass_unions_external_and_in_document_style(tmp_path):
    doc_html = "<html><body><style>.aside { font-style: italic }</style><p>x</p></body></html>"
    path = _build_epub(
        tmp_path / "union.epub",
        css_content=".thought { font-style: oblique }",
        doc_html=doc_html,
    )
    book = epub.read_epub(str(path), options={"ignore_ncx": True})
    m = _build_italic_style_map(book)
    assert {"aside", "thought"} <= m.italic_classes
