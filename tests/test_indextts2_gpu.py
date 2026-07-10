"""Gated IndexTTS-2 smoke test: drives the REAL worker + REAL model in the separate cu128 env.

Deselected by default (``-m 'not gpu'``) and skipped even under ``-m gpu`` unless
``indextts2_worker_python`` + ``indextts2_checkpoints_dir`` are configured and present — they are
absent on CI and a fresh box, and the model only exists in the separate cu128 env. Run it on the
configured GPU box with:

    uv run pytest -m gpu -k indextts2_real

Optionally point ``INDEXTTS2_TEST_REFERENCE`` at a real speech clip; otherwise a short synthetic
tone is used, which is enough to prove the worker/adapter plumbing end to end.
"""

import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from seiyuu.engines import get_engine
from seiyuu.settings import get_settings

pytestmark = pytest.mark.gpu


def _require_configured():
    cfg = get_settings()
    if not cfg.indextts2_worker_python or not Path(cfg.indextts2_worker_python).exists():
        pytest.skip("indextts2_worker_python not set to the cu128 env python; skipping GPU smoke")
    if not cfg.indextts2_checkpoints_dir or not Path(cfg.indextts2_checkpoints_dir).is_dir():
        pytest.skip("indextts2_checkpoints_dir not set / weights missing; skipping GPU smoke")
    return cfg


def _reference(tmp_path: Path) -> Path:
    # explicit env var wins (one-off runs), then the configured clip (settings/.env), then a
    # synthetic tone (proves plumbing, not voice quality)
    env = os.environ.get("INDEXTTS2_TEST_REFERENCE")
    if env and Path(env).is_file():
        return Path(env)
    configured = get_settings().indextts2_test_reference
    if configured and Path(configured).is_file():
        return Path(configured)
    ref = tmp_path / "ref.wav"
    t = np.linspace(0, 3.0, int(3.0 * 22050), endpoint=False)
    sf.write(ref, (0.3 * np.sin(2 * np.pi * 140 * t)).astype(np.float32), 22050, subtype="PCM_16")
    return ref


def test_indextts2_real_render_smoke(tmp_path):
    """Real end-to-end: subprocess worker loads the model, synthesizes a segment, and the audio
    comes back canonical 24 kHz. Also checks seed reproducibility (the SegmentKey contract)."""
    _require_configured()
    voices = tmp_path / "voices"
    (voices / "smoke_voice").mkdir(parents=True)
    shutil.copyfile(_reference(tmp_path), voices / "smoke_voice" / "reference.wav")

    engine = get_engine("indextts2", voices_dir=voices)  # real worker; rest of config from settings
    try:
        text = "This is a short IndexTTS 2 smoke test."
        audio = engine.synthesize(text, "smoke_voice", {"seed": 41172})
        assert audio.sample_rate == 24_000  # to_canonical resampled from 22050
        assert audio.samples.size > 0 and float(np.abs(audio.samples).max()) > 0.0

        # identical seed must reproduce identical audio, or the frozen SegmentKey would map one
        # key to different renders (same discipline the CPU test asserts via a fake worker).
        again = engine.synthesize(text, "smoke_voice", {"seed": 41172})
        assert np.array_equal(audio.samples, again.samples)
    finally:
        engine.unload()  # terminate the worker -> reclaim VRAM
