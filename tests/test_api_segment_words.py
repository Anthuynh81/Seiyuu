"""F2 — GET /books/{id}/segments/{block_id}/words: resolves the wav via the manifest exactly
like /audio, computes-then-caches on first hit, serves the cached SegmentWords thereafter, and
404s on scene-break / out-of-range / missing wav / unknown block. The shared CPU aligner on
app.state is replaced with a fake so the suite stays offline (no live whisper).
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from seiyuu.api.main import create_app
from seiyuu.engines import AudioFile
from seiyuu.render.cache import SegmentKey
from seiyuu.render.models import RenderedChapter, RenderedSegment, RenderManifest
from seiyuu.settings import Settings
from seiyuu.validate import WordTiming


def make_settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        books_dir=tmp_path / "books",
        output_dir=tmp_path / "output",
        voices_dir=tmp_path / "voices",
        data_dir=tmp_path / "data",
        anthropic_api_key=None,
        elevenlabs_api_key=None,
    )


class FakeAligner:
    def __init__(self) -> None:
        self.calls = 0

    def transcribe_words(self, wav_path):
        self.calls += 1
        return [
            WordTiming(start=0.0, end=0.3, word=" Hello"),
            WordTiming(start=0.3, end=0.6, word=" there"),
        ]


@pytest.fixture
def cfg(tmp_path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
def client(cfg):
    app = create_app(settings=cfg)
    with TestClient(app) as c:
        c.app = app
        c.app.state.aligner = FakeAligner()  # offline: no live whisper
        yield c


def _seed_render(cfg: Settings, book_id: str) -> str:
    """A manifest with one real speakable wav + a scene break, mirroring segment layout."""
    o = cfg.output_dir / book_id
    cache = o / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    key = SegmentKey.build(
        engine="kokoro",
        engine_model_version="kokoro-0.9.4",
        voice_id="af_heart",
        settings={"speed": 1.0},
        seed=1,
        normalized_text="Hello there.",
    )
    stem = key.key_hash
    AudioFile(samples=np.zeros(12_000, dtype=np.float32)).save(cache / f"{stem}.wav")
    segs = [
        RenderedSegment(
            block_id="ch001_b0001",
            type="paragraph",
            wav=f"cache/{stem}.wav",
            duration_seconds=0.5,
            voice_id="af_heart",
        ),
        RenderedSegment(block_id="ch001_b0002", type="scene_break"),
    ]
    manifest = RenderManifest(
        book_id=book_id,
        engine="kokoro",
        engine_model_version="kokoro-0.9.4",
        voice_id="af_heart",
        seed=1,
        chapters=[RenderedChapter(index=1, title="C1", segments=segs)],
    )
    (o / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    return stem


def test_words_computes_then_caches(client, cfg):
    _seed_render(cfg, "bk")
    aligner: FakeAligner = client.app.state.aligner

    r1 = client.get("/api/books/bk/segments/ch001_b0001/words")
    assert r1.status_code == 200, r1.text
    body = r1.json()
    assert [w["word"] for w in body["words"]] == [" Hello", " there"]
    assert body["audio_duration"] == pytest.approx(0.5, abs=0.01)
    assert body["source"] == "whisper"
    assert aligner.calls == 1
    # sidecar now on disk
    assert list((cfg.output_dir / "bk" / "cache").glob("*.words.json"))

    r2 = client.get("/api/books/bk/segments/ch001_b0001/words")
    assert r2.status_code == 200
    assert r2.json() == body
    assert aligner.calls == 1  # served from cache, not re-transcribed


def test_words_404_on_scene_break(client, cfg):
    _seed_render(cfg, "bk")
    r = client.get("/api/books/bk/segments/ch001_b0002/words")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"


def test_words_404_on_unknown_block(client, cfg):
    _seed_render(cfg, "bk")
    r = client.get("/api/books/bk/segments/nope_b9999/words")
    assert r.status_code == 404


def test_words_404_on_out_of_range_segment(client, cfg):
    _seed_render(cfg, "bk")
    r = client.get("/api/books/bk/segments/ch001_b0001/words", params={"segment": 5})
    assert r.status_code == 404


def test_words_404_when_wav_missing_on_disk(client, cfg):
    stem = _seed_render(cfg, "bk")
    (cfg.output_dir / "bk" / "cache" / f"{stem}.wav").unlink()
    r = client.get("/api/books/bk/segments/ch001_b0001/words")
    assert r.status_code == 404


def test_words_404_for_unrendered_book(client, cfg):
    # book that exists (ingested) but has no manifest -> not_found, like the audio route
    (cfg.books_dir / "bk").mkdir(parents=True)
    (cfg.books_dir / "bk" / "normalized.json").write_text(
        '{"book_meta": {"title": "T", "authors": []}, "chapters": []}', encoding="utf-8"
    )
    r = client.get("/api/books/bk/segments/ch001_b0001/words")
    assert r.status_code == 404


def test_chapter_words_batch(client, cfg):
    # One request returns every audio-bearing clip of the chapter, keyed like the player
    # clips; scene breaks are absent, and a wav missing on disk is omitted (client falls
    # back to interpolation) instead of failing the whole page.
    stem = _seed_render(cfg, "bk")
    aligner: FakeAligner = client.app.state.aligner

    r = client.get("/api/books/bk/chapters/1/words")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["chapter"] == 1
    assert set(body["words"]) == {"ch001_b0001:0"}
    assert [w["word"] for w in body["words"]["ch001_b0001:0"]["words"]] == [" Hello", " there"]
    assert aligner.calls == 1

    # served from the sidecar cache on the second request; per-clip endpoint shares it
    assert client.get("/api/books/bk/chapters/1/words").status_code == 200
    assert client.get("/api/books/bk/segments/ch001_b0001/words").status_code == 200
    assert aligner.calls == 1

    assert client.get("/api/books/bk/chapters/9/words").status_code == 404

    (cfg.output_dir / "bk" / "cache" / f"{stem}.wav").unlink()
    gone = client.get("/api/books/bk/chapters/1/words")
    assert gone.status_code == 200
    assert gone.json()["words"] == {}
