"""Theme B — Emotion UX (v1.1): F2a per-segment emotion-override edit op + F2b per-render
apply_emotion toggle. Fixtures only, no live LLM/TTS.

F2a proves the emotion override rides the SAME durable edit overlay the speaker edits use
(set/clear, replayed onto the non-frozen segment_emotions), and is seen IDENTICALLY by render
and the cost estimate. F2b proves the per-render apply_emotion override resolves against the
server default and is honored with parity at render AND estimate (same emotion-folded keys the
cost gate authorizes), while OFF stays byte-identical to a no-emotion render.
"""

import pytest
from fastapi.testclient import TestClient

from factories import make_book
from fake_engine import FakeEngine
from seiyuu.api.main import create_app
from seiyuu.api.money import compute_estimate, effective_apply_emotion
from seiyuu.attribute import ATTRIBUTION_NAME
from seiyuu.attribute.models import (
    AttributedChapter,
    AttributionReport,
    Character,
    CharacterRegistry,
    EmotionLabel,
    EmotionVerdict,
    Segment,
    SegmentType,
)
from seiyuu.gpu import GpuResourceManager
from seiyuu.render import estimate_render_cost, render_book_multivoice
from seiyuu.services import save_assignment
from seiyuu.services.edits import EditLog, SetEmotion, anchor_op, apply_edits
from seiyuu.settings import Settings
from seiyuu.voices import VoiceAssignment, VoiceLibrary, VoiceMeta
from seiyuu.voices.models import VoiceKind
from test_api_m6b1 import make_settings

_BOOK_ID = "test-book-00000000"

_DIALOGUE_BLOCKS = {1: "ch001_b0002", 2: "ch002_b0002"}


def _report(with_emotion: bool) -> AttributionReport:
    """A 2-chapter report; alice speaks one chatterbox dialogue per chapter (ch1 also has a
    trailing narration paragraph). Mirrors factories.make_book's block layout."""

    def chapter(idx: int) -> AttributedChapter:
        title = f"Chapter {idx}"
        segs = [
            Segment(block_id=f"ch{idx:03d}_b0001", type=SegmentType.NARRATION, text=title),
            Segment(
                block_id=_DIALOGUE_BLOCKS[idx],
                type=SegmentType.DIALOGUE,
                text="Hello world." if idx == 1 else "Second chapter.",
                speaker="alice",
            ),
        ]
        emotions = [
            None,
            EmotionVerdict(label=EmotionLabel.ANGRY, intensity=3) if with_emotion else None,
        ]
        if idx == 1:
            segs.append(Segment(block_id="ch001_b0004", type=SegmentType.NARRATION, text="After."))
            emotions.append(None)
        return AttributedChapter(index=idx, title=title, segments=segs, segment_emotions=emotions)

    return AttributionReport(
        book_id=_BOOK_ID,
        provider_id="local",
        model_id="qwen2.5:7b",
        prompt_version="v5",
        registry=CharacterRegistry(characters=[Character(id="alice", canonical_name="Alice")]),
        chapters=[chapter(1), chapter(2)],
    )


def _library(voices_dir) -> VoiceLibrary:
    lib = VoiceLibrary(voices_dir)
    lib.save(
        VoiceMeta(voice_id="narrator_v", name="N", kind=VoiceKind.PRESET,
                  engine="kokoro", preset_id="af_heart")
    )  # fmt: skip
    lib.save(
        VoiceMeta(voice_id="alice_v", name="Alice", kind=VoiceKind.PRESET,
                  engine="chatterbox", preset_id="clone_alice")
    )  # fmt: skip
    return lib


def _assignment() -> VoiceAssignment:
    return VoiceAssignment(
        book_id=_BOOK_ID, narrator_voice_id="narrator_v", assignments={"alice": "alice_v"}
    )


def _patch(monkeypatch) -> None:
    monkeypatch.setattr("seiyuu.render.pipeline.get_engine", lambda engine_id, **kw: FakeEngine())


# =====================================================================================
# F2a — emotion-override edit op (overlay onto segment_emotions)
# =====================================================================================


def test_set_emotion_overlay_sets_then_clears():
    base = _report(with_emotion=False)  # segment_emotions all None
    # SET the ch1 dialogue (index 0 within its block) to happy/2.
    set_op = anchor_op(
        base,
        SetEmotion(
            block_id="ch001_b0002",
            segment_index=0,
            emotion=EmotionVerdict(label=EmotionLabel.HAPPY, intensity=2),
        ),
    )
    assert set_op.text_anchor  # server-filled content anchor, exactly like reassign
    effective, warns = apply_edits(base, EditLog(ops=[set_op]))
    assert warns == []
    # dialogue is the flat index-1 segment of ch1; the overlay lands there, others stay None
    ch1 = effective.chapters[0]
    assert ch1.segment_emotions[1] == EmotionVerdict(label=EmotionLabel.HAPPY, intensity=2)
    assert ch1.segment_emotions[0] is None and ch1.segment_emotions[2] is None
    # the input report is never mutated (apply_edits deep-copies)
    assert base.chapters[0].segment_emotions[1] is None

    # CLEAR it back with a second op (emotion=None).
    clear_op = anchor_op(base, SetEmotion(block_id="ch001_b0002", segment_index=0, emotion=None))
    cleared, warns2 = apply_edits(base, EditLog(ops=[set_op, clear_op]))
    assert warns2 == []
    assert cleared.chapters[0].segment_emotions[1] is None


def test_set_emotion_overlay_normalizes_v3_report():
    """A v3/v4 chapter carries an EMPTY segment_emotions; the overlay must normalize it to a
    full-length list before writing, so the render/estimate index-alignment can't desync."""
    base = _report(with_emotion=False)
    base.chapters[0].segment_emotions = []  # pre-emotion shape
    op = anchor_op(
        base,
        SetEmotion(
            block_id="ch001_b0002",
            segment_index=0,
            emotion=EmotionVerdict(label=EmotionLabel.SAD, intensity=1),
        ),
    )
    effective, warns = apply_edits(base, EditLog(ops=[op]))
    assert warns == []
    emotions = effective.chapters[0].segment_emotions
    assert len(emotions) == len(effective.chapters[0].segments)  # normalized to full length
    assert emotions[1] == EmotionVerdict(label=EmotionLabel.SAD, intensity=1)


def test_set_emotion_overlay_render_and_estimate_parity(tmp_path, monkeypatch):
    """The overlaid emotion is folded into settings_hash IDENTICALLY at render and estimate."""
    _patch(monkeypatch)
    lib, assign, out = _library(tmp_path / "voices"), _assignment(), tmp_path / "out"
    base = _report(with_emotion=False)
    ops = [
        anchor_op(
            base,
            SetEmotion(
                block_id=_DIALOGUE_BLOCKS[i],
                segment_index=0,
                emotion=EmotionVerdict(label=EmotionLabel.ANGRY, intensity=3),
            ),
        )  # fmt: skip
        for i in (1, 2)
    ]
    effective, warns = apply_edits(base, EditLog(ops=ops))
    assert warns == []

    render_book_multivoice(
        effective, make_book(), lib, assign, out, gpu=GpuResourceManager(), apply_emotion=True
    )
    # estimate the SAME overlaid report with apply_emotion=True -> every key already cached (parity)
    on = estimate_render_cost(effective, make_book(), lib, assign, out, apply_emotion=True)
    assert on.free_segments == 0
    # apply_emotion=False rebuilds the two chatterbox dialogue keys WITHOUT the folded emotion,
    # so they miss the emotion-keyed cache — proof the overlay emotion rode settings_hash.
    off = estimate_render_cost(effective, make_book(), lib, assign, out, apply_emotion=False)
    assert off.free_segments == 2


def test_old_edit_logs_without_set_emotion_still_parse():
    # a pre-F2a log (only reassign/rename) parses unchanged (additive union member)
    old = EditLog.model_validate(
        {
            "version": 1,
            "ops": [
                {"op": "reassign", "block_id": "ch001_b0002", "segment_index": 0, "speaker": None}
            ],
        }  # fmt: skip
    )
    assert old.ops[0].op == "reassign"
    # and a NEW log with a set_emotion op round-trips
    new = EditLog.model_validate_json(
        EditLog(
            ops=[
                SetEmotion(
                    block_id="ch001_b0002",
                    segment_index=0,
                    emotion=EmotionVerdict(label=EmotionLabel.TENSE, intensity=2),
                )
            ]
        ).model_dump_json()  # fmt: skip
    )
    assert new.ops[0].op == "set_emotion"
    assert new.ops[0].emotion.label == EmotionLabel.TENSE


# =====================================================================================
# F2b — per-render apply_emotion toggle (override resolves against the server default)
# =====================================================================================


def test_effective_apply_emotion_resolution():
    off = Settings(_env_file=None, apply_emotion=False)
    on = Settings(_env_file=None, apply_emotion=True)
    assert effective_apply_emotion(off, None) is False  # None -> cfg default
    assert effective_apply_emotion(on, None) is True
    assert effective_apply_emotion(off, True) is True  # explicit override wins both ways
    assert effective_apply_emotion(on, False) is False


def _stage_render_book(
    cfg: Settings, report: AttributionReport
) -> tuple[VoiceLibrary, VoiceAssignment]:
    book_dir = cfg.books_dir / _BOOK_ID
    book_dir.mkdir(parents=True, exist_ok=True)
    (book_dir / ATTRIBUTION_NAME).write_text(report.model_dump_json(), encoding="utf-8")
    lib = _library(cfg.voices_dir)
    assign = _assignment()
    save_assignment(assign, cfg.output_dir)
    return lib, assign


def test_per_render_override_applies_when_cfg_default_off(tmp_path, monkeypatch):
    """cfg.apply_emotion is OFF, but a per-render override of True must price AND render the
    emotion-folded keys identically (parity) — the gate authorizes exactly what render bills."""
    _patch(monkeypatch)
    cfg = make_settings(tmp_path, apply_emotion=False)
    report = _report(with_emotion=True)
    lib, assign = _stage_render_book(cfg, report)
    book, out = make_book(), cfg.output_dir / _BOOK_ID

    # Render this book with the per-render override ON (even though cfg default is OFF).
    render_book_multivoice(
        report, book, lib, assign, out,
        gpu=GpuResourceManager(),
        apply_emotion=effective_apply_emotion(cfg, True),
    )  # fmt: skip

    # The estimate honoring the SAME override -> every emotion-folded key is cached (parity).
    ctx_on = compute_estimate(
        cfg, None, book, _BOOK_ID, mode="multivoice", chapters=(), single=None, apply_emotion=True
    )
    assert ctx_on.est.free_segments == 0

    # No override (None) -> falls back to cfg's OFF default, so the two chatterbox dialogue keys
    # lack the folded emotion and miss the cache the emotion-on render filled.
    ctx_none = compute_estimate(
        cfg, None, book, _BOOK_ID, mode="multivoice", chapters=(), single=None, apply_emotion=None
    )
    assert ctx_none.est.free_segments == 2


def test_apply_emotion_off_is_segmentkey_byte_identical(tmp_path, monkeypatch):
    """cfg default ON, but a per-render override of False must keep the SegmentKey byte-identical
    to a no-emotion render (no cache churn)."""
    _patch(monkeypatch)
    cfg = make_settings(tmp_path, apply_emotion=True)
    # Render a PLAIN (no-emotion) report with emotion OFF...
    plain = _report(with_emotion=False)
    lib, assign = _stage_render_book(cfg, plain)
    book, out = make_book(), cfg.output_dir / _BOOK_ID
    render_book_multivoice(
        plain, book, lib, assign, out,
        gpu=GpuResourceManager(), apply_emotion=effective_apply_emotion(cfg, False),
    )  # fmt: skip
    # ...now an EMOTION-tagged report estimated with the override OFF is fully cached: OFF ignores
    # segment_emotions entirely, so the keys are byte-identical to the plain render.
    (cfg.books_dir / _BOOK_ID / ATTRIBUTION_NAME).write_text(
        _report(with_emotion=True).model_dump_json(), encoding="utf-8"
    )
    ctx = compute_estimate(
        cfg, None, book, _BOOK_ID, mode="multivoice", chapters=(), single=None, apply_emotion=False
    )
    assert ctx.est.cached_segments > 0 and ctx.est.free_segments == 0


# =====================================================================================
# F2b — SystemStatus / settings expose the server default
# =====================================================================================


@pytest.mark.parametrize("default", [False, True])
def test_system_and_settings_report_apply_emotion_default(tmp_path, default):
    app = create_app(settings=make_settings(tmp_path, apply_emotion=default))
    with TestClient(app) as c:
        assert c.get("/api/system").json()["apply_emotion"] is default
        assert c.get("/api/settings").json()["apply_emotion"] is default
