"""Unit tests for the dependency-light CSS italic parser (THOUGHT Phase 2a).

The parser's only sanctioned error direction is UNDER-capture ("miss an italic"), never
over-capture. These pin classification, last-rule-wins, comment stripping, @-rule body
skipping, bare/universal-selector rejection, cross-sheet normal subtraction, keyword
case-insensitivity, verbatim class-token case, and the documented ``font:`` shorthand miss.
"""

from seiyuu.ingest.css_italics import ItalicStyleMap, parse_css_italics


def _classes(*sheets: str) -> set[str]:
    return set(parse_css_italics(sheets).italic_classes)


def test_italic_and_oblique_classify_as_italic_normal_does_not():
    m = parse_css_italics(
        [
            ".thought { font-style: italic }"
            " .aside { font-style: oblique }"
            " .plain { font-style: normal }"
        ]
    )
    assert m.italic_classes == frozenset({"thought", "aside"})


def test_last_font_style_in_a_block_wins():
    # In-block cascade: the later declaration overrides the earlier one.
    assert _classes(".x { font-style: italic; font-style: normal }") == set()
    assert _classes(".y { font-style: normal; font-style: italic }") == {"y"}


def test_comments_are_stripped_before_parsing():
    css = "/* .ghost { font-style: italic } */ .real { font-style: italic }"
    assert _classes(css) == {"real"}


def test_multiline_comment_stripped():
    css = (
        ".a {\n  /* font-style: italic;\n     spanning lines */\n"
        "  font-style: normal;\n}\n.b { font-style: italic }"
    )
    assert _classes(css) == {"b"}


def test_at_media_body_is_skipped_whole():
    css = "@media screen { .m { font-style: italic } } .out { font-style: italic }"
    m = parse_css_italics([css])
    assert m.italic_classes == frozenset({"out"})  # .m inside @media is a deliberate miss


def test_at_font_face_and_keyframes_bodies_skipped():
    css = (
        "@font-face { font-family: X; font-style: italic }"
        "@keyframes k { from { font-style: italic } to { font-style: normal } }"
        ".real { font-style: italic }"
    )
    assert _classes(css) == {"real"}


def test_bare_element_and_universal_selectors_are_dropped():
    css = (
        "p { font-style: italic } div { font-style: italic }"
        " * { font-style: italic } body { font-style: italic }"
    )
    assert _classes(css) == set()  # no class token -> nothing harvested


def test_element_qualified_class_still_harvests_the_class():
    # p.italic keeps the class token; the bare element part contributes nothing on its own.
    assert _classes("p.thought { font-style: italic }") == {"thought"}


def test_normal_forced_class_subtracted_across_sheets():
    # Sheet A italicizes .x; sheet B resets it to normal -> excluded (precision over recall).
    assert _classes(".x { font-style: italic }", ".x { font-style: normal }") == set()
    # Order-independent: normal in the first sheet still wins.
    assert _classes(".x { font-style: normal }", ".x { font-style: italic }") == set()


def test_keywords_are_case_insensitive():
    assert _classes(".x { FONT-STYLE: ITALIC }") == {"x"}
    assert _classes(".y { Font-Style: Oblique }") == {"y"}
    assert _classes(".z { font-style: ITALIC }", ".z { font-style: NORMAL }") == set()


def test_class_tokens_are_verbatim_case():
    m = parse_css_italics([".Thought { font-style: italic } .THOUGHT { font-style: italic }"])
    assert m.italic_classes == frozenset({"Thought", "THOUGHT"})
    assert "thought" not in m.italic_classes


def test_font_shorthand_is_a_documented_miss():
    # The `font:` shorthand carries the style but we only parse the `font-style` longhand.
    assert _classes(".x { font: italic 12px serif }") == set()


def test_grouped_selectors_all_get_the_class():
    assert _classes(".a, .b, .c { font-style: italic }") == {"a", "b", "c"}


def test_block_without_font_style_contributes_nothing():
    assert _classes(".x { color: red; font-weight: bold }") == set()


def test_em_i_forced_normal_flags_recorded_but_tokens_are_element_not_class():
    m = parse_css_italics(["em { font-style: normal } i { font-style: normal }"])
    assert m.em_forced_normal is True
    assert m.i_forced_normal is True
    assert m.italic_classes == frozenset()


def test_em_i_flags_default_false_and_class_selectors_do_not_trip_them():
    m = parse_css_italics([".important { font-style: normal }"])
    assert m.em_forced_normal is False
    assert m.i_forced_normal is False


def test_empty_and_whitespace_sheets_yield_empty_map():
    assert parse_css_italics([]) == ItalicStyleMap()
    assert parse_css_italics(["", "   \n\t "]) == ItalicStyleMap()


def test_stray_braces_do_not_crash_or_leak():
    assert _classes("} .x { font-style: italic } }") == {"x"}
