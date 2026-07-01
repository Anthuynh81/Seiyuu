"""Repository layer: atomic writes + the book registry (M6a commit 1)."""

import json

import pytest

from seiyuu.repository import (
    RepositoryError,
    atomic_write_bytes,
    atomic_write_text,
    get_book_status,
    list_books,
    resolve_book_id,
)
from seiyuu.repository import atomic as atomic_mod
from seiyuu.repository import books as books_mod

# --- atomic writes ---


def test_atomic_write_text_roundtrip(tmp_path):
    p = tmp_path / "sub" / "a.json"  # parent does not exist yet
    assert atomic_write_text(p, '{"x": 1}') == p
    assert p.read_text(encoding="utf-8") == '{"x": 1}'


def test_atomic_write_bytes_roundtrip(tmp_path):
    p = tmp_path / "b.bin"
    atomic_write_bytes(p, b"\x00\x01\x02")
    assert p.read_bytes() == b"\x00\x01\x02"


def test_atomic_write_overwrites(tmp_path):
    p = tmp_path / "a.txt"
    atomic_write_text(p, "old")
    atomic_write_text(p, "new")
    assert p.read_text() == "new"


def test_atomic_write_leaves_no_temp_file(tmp_path):
    atomic_write_text(tmp_path / "a.txt", "hi")
    assert [x.name for x in tmp_path.iterdir()] == ["a.txt"]


def test_atomic_write_failure_preserves_original_and_cleans_temp(tmp_path, monkeypatch):
    p = tmp_path / "a.txt"
    atomic_write_text(p, "original")

    def boom(src, dst):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(atomic_mod.os, "replace", boom)
    with pytest.raises(OSError, match="simulated replace failure"):
        atomic_write_text(p, "would-corrupt")
    assert p.read_text() == "original"  # destination is untouched by a failed write
    assert [x.name for x in tmp_path.iterdir()] == ["a.txt"]  # no orphan temp left behind


# --- book registry ---


def _scaffold_normalized(books_dir, book_id, *, title="T", authors=("A",)):
    p = books_dir / book_id / "normalized.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"book_meta": {"title": title, "authors": list(authors)}, "chapters": []}
    p.write_text(json.dumps(payload), encoding="utf-8")


def test_list_books_empty_when_roots_absent(tmp_path):
    assert list_books(books_dir=tmp_path / "books", output_dir=tmp_path / "output") == []


def test_book_status_progresses_through_stages(tmp_path):
    books, output = tmp_path / "books", tmp_path / "output"
    bid = "my-book-abcd1234"
    _scaffold_normalized(books, bid, title="My Book", authors=["Jane"])

    st = get_book_status(bid, books_dir=books, output_dir=output)
    assert st.title == "My Book" and st.authors == ["Jane"]
    assert st.ingested
    assert not (st.attributed or st.assigned or st.rendered or st.assembled or st.mastered)

    (books / bid / "attribution.json").write_text("{}", encoding="utf-8")
    odir = output / bid
    odir.mkdir(parents=True, exist_ok=True)
    (odir / "assignments.json").write_text("{}", encoding="utf-8")
    (odir / "manifest.json").write_text("{}", encoding="utf-8")
    (odir / "chapters").mkdir()
    (odir / "chapters" / "ch001.mp3").write_bytes(b"x")
    (odir / f"{bid}.m4b").write_bytes(b"x")

    st2 = get_book_status(bid, books_dir=books, output_dir=output)
    assert st2.ingested and st2.attributed and st2.assigned
    assert st2.rendered and st2.assembled and st2.mastered


def test_assembled_requires_an_mp3_not_just_the_dir(tmp_path):
    books, output = tmp_path / "books", tmp_path / "output"
    bid = "b-1"
    _scaffold_normalized(books, bid)
    (output / bid / "chapters").mkdir(parents=True, exist_ok=True)  # empty
    assert not get_book_status(bid, books_dir=books, output_dir=output).assembled


def test_list_books_discovers_and_sorts(tmp_path):
    books, output = tmp_path / "books", tmp_path / "output"
    _scaffold_normalized(books, "zebra-0001")
    _scaffold_normalized(books, "alpha-0002")
    ids = [b.book_id for b in list_books(books_dir=books, output_dir=output)]
    assert ids == ["alpha-0002", "zebra-0001"]


def test_list_books_includes_output_only_book(tmp_path):
    # A book whose books/ dir was removed but whose renders remain must still be listed.
    books, output = tmp_path / "books", tmp_path / "output"
    (output / "ghost-9999").mkdir(parents=True, exist_ok=True)
    (output / "ghost-9999" / "manifest.json").write_text("{}", encoding="utf-8")
    by_id = {b.book_id: b for b in list_books(books_dir=books, output_dir=output)}
    assert "ghost-9999" in by_id
    assert by_id["ghost-9999"].rendered and not by_id["ghost-9999"].ingested
    assert by_id["ghost-9999"].title is None


def test_resolve_book_id_exact_and_prefix(tmp_path):
    books, output = tmp_path / "books", tmp_path / "output"
    _scaffold_normalized(books, "pride-abcd1234")
    assert resolve_book_id("pride-abcd1234", books_dir=books, output_dir=output) == "pride-abcd1234"
    assert resolve_book_id("pride", books_dir=books, output_dir=output) == "pride-abcd1234"


def test_resolve_book_id_not_found(tmp_path):
    with pytest.raises(RepositoryError, match="not found"):
        resolve_book_id("nope", books_dir=tmp_path / "books", output_dir=tmp_path / "output")


def test_resolve_book_id_ambiguous(tmp_path):
    books, output = tmp_path / "books", tmp_path / "output"
    _scaffold_normalized(books, "book-1111")
    _scaffold_normalized(books, "book-2222")
    with pytest.raises(RepositoryError, match="ambiguous"):
        resolve_book_id("book", books_dir=books, output_dir=output)


def test_marker_names_stay_in_sync_with_stage_constants():
    # The registry keeps local marker literals to stay import-light; guard against drift.
    from seiyuu.attribute import ATTRIBUTION_NAME
    from seiyuu.render import MANIFEST_NAME
    from seiyuu.voices import ASSIGNMENT_NAME

    assert books_mod.ATTRIBUTION_NAME == ATTRIBUTION_NAME
    assert books_mod.MANIFEST_NAME == MANIFEST_NAME
    assert books_mod.ASSIGNMENT_NAME == ASSIGNMENT_NAME
    assert books_mod.NORMALIZED_NAME == "normalized.json"
