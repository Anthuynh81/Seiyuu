"""Gated Breeze TTS 2 smoke test: drives the REAL worker + REAL model in the separate Breeze env.

Deselected by default (``-m 'not gpu'``) and skipped even under ``-m gpu`` unless
``BREEZE_WORKER_PYTHON``, ``BREEZE_REPO_DIR`` and ``BREEZE_CHECKPOINT_DIR`` are configured and
present — they are absent on CI and a fresh box. Run it on the configured GPU box with:

    uv run pytest -m gpu -k breeze_real -s

It takes a few minutes: a 7.7 GB load, then the fast path's compile (16-50 s the first time),
then three short lines. Breeze clones from a clip AND its exact transcript, so a synthetic tone
won't do: the test borrows the first voice in the library that has ``reference.txt`` beside its
``reference.wav`` (a designed voice whose kept take passed the speech check does), or the clip
named by the ``BREEZE_TEST_REFERENCE`` environment variable (with its transcript in a ``.txt`` of
the same name). Set ``BREEZE_SMOKE_OUT`` to a directory to keep the three WAVs for a listen.
"""

import hashlib
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pytest

from seiyuu.engines import get_engine
from seiyuu.engines.breeze_engine import checkpoint_present, repo_present
from seiyuu.engines.qwen3tts_engine import REF_TEXT_DIGEST_KEY, reference_text_digest
from seiyuu.settings import get_settings

pytestmark = pytest.mark.gpu

SEED = 41172
LINE = "I counted the doors twice, and there was one more than there should have been."
INSTRUCTION = "Calm, dry and quiet, a little lower and more intimate than speaking out loud."


def _require_configured():
    cfg = get_settings()
    if not cfg.breeze_worker_python or not Path(cfg.breeze_worker_python).exists():
        pytest.skip("BREEZE_WORKER_PYTHON not set to the Breeze env python; skipping GPU smoke")
    if not repo_present(cfg.breeze_repo_dir):
        pytest.skip("BREEZE_REPO_DIR is not a breeze-tts checkout; skipping GPU smoke")
    if not checkpoint_present(cfg.breeze_checkpoint_dir):
        pytest.skip("BREEZE_CHECKPOINT_DIR not set / weights missing; skipping GPU smoke")
    return cfg


def _reference(cfg) -> tuple[Path, str]:
    """A real speech clip and its exact transcript."""
    env = os.environ.get("BREEZE_TEST_REFERENCE")
    if env:
        clip, txt = Path(env), Path(env).with_suffix(".txt")
        if not clip.is_file() or not txt.is_file() or not txt.read_text(encoding="utf-8").strip():
            pytest.skip(f"BREEZE_TEST_REFERENCE: {clip} or its transcript {txt.name} is missing")
        return clip, txt.read_text(encoding="utf-8").strip()
    voices_dir = Path(cfg.voices_dir)
    for voice in sorted(voices_dir.iterdir()) if voices_dir.is_dir() else []:
        wav, txt = voice / "reference.wav", voice / "reference.txt"
        if wav.is_file() and txt.is_file() and txt.read_text(encoding="utf-8").strip():
            return wav, txt.read_text(encoding="utf-8").strip()
    pytest.skip("no voice with reference.wav + reference.txt to clone; skipping GPU smoke")


def test_breeze_real_render_smoke(tmp_path):
    """Real end to end: the worker loads the model, warms the fast path, clones a line, repeats
    it from the same seed, and speaks it again with a delivery instruction."""
    cfg = _require_configured()
    wav, transcript = _reference(cfg)
    voices = tmp_path / "voices"
    (voices / "smoke_voice").mkdir(parents=True)
    shutil.copyfile(wav, voices / "smoke_voice" / "reference.wav")
    (voices / "smoke_voice" / "reference.txt").write_text(transcript + "\n", encoding="utf-8")
    sha = hashlib.sha256(wav.read_bytes()).hexdigest()
    settings = {"seed": SEED, REF_TEXT_DIGEST_KEY: reference_text_digest(sha, transcript)}

    engine = get_engine("breeze", voices_dir=voices)  # real worker; rest of config from settings
    try:
        started = time.perf_counter()
        engine.warm()
        loaded = time.perf_counter() - started

        started = time.perf_counter()
        clone = engine.synthesize(LINE, "smoke_voice", settings)
        clone_wall = time.perf_counter() - started
        assert clone.sample_rate == 24_000
        assert float(np.abs(clone.samples).max()) > 0.0
        # 78 characters of speech: seconds, not a click and not a runaway (the cap is 19 s)
        assert 2.0 < clone.duration_seconds < 15.0

        # Within one worker the same seed must give the same audio (the seed plumbing works and
        # nothing between two lines shifts the sampling). A fresh worker recaptures its graphs
        # and re-encodes the clip, which this does not cover.
        again = engine.synthesize(LINE, "smoke_voice", settings)
        assert np.array_equal(clone.samples, again.samples)

        started = time.perf_counter()
        directed = engine.synthesize(LINE, "smoke_voice", {**settings, "instruction": INSTRUCTION})
        directed_wall = time.perf_counter() - started
        assert directed.duration_seconds > 2.0  # a runaway is caught by capped_lines below
        assert not np.array_equal(clone.samples, directed.samples)  # the instruction did something

        assert engine.worker_restarts == 0 and engine.capped_lines == 0
        clone_s, directed_s = clone.duration_seconds, directed.duration_seconds
        print(
            f"\nbreeze smoke ({cfg.breeze_fast}): load+warm {loaded:.0f}s | "
            f"clone {clone_s:.1f}s audio, RTF {clone_wall / clone_s:.2f} | "
            f"direction {directed_s:.1f}s audio, RTF {directed_wall / directed_s:.2f}"
        )
        out = os.environ.get("BREEZE_SMOKE_OUT")
        if out:
            Path(out).mkdir(parents=True, exist_ok=True)
            clone.save(Path(out) / "breeze_clone.wav")
            again.save(Path(out) / "breeze_clone_again.wav")
            directed.save(Path(out) / "breeze_direction.wav")
    finally:
        engine.unload()  # terminate the worker -> reclaim VRAM
