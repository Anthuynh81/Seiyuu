"""Real Chatterbox smoke + determinism (GATED: @pytest.mark.gpu, excluded by default).

Run with `uv run pytest -m gpu tests/test_chatterbox_gpu.py`. The FIRST run downloads
multi-GB weights to the HF cache — ask before running. This is the test that answers the
open question of whether torch.manual_seed alone makes generate() bit-reproducible enough
to trust the frozen segment cache for cloned voices.
"""

import numpy as np
import pytest
import soundfile as sf
import torch

pytestmark = pytest.mark.gpu


def _write_reference(path, seconds=8.0, sr=24_000):
    t = np.arange(int(seconds * sr)) / sr
    tone = 0.2 * np.sin(2 * np.pi * 150 * t).astype(np.float32)
    sf.write(str(path), tone, sr, subtype="PCM_16")


def test_chatterbox_clone_and_seed_determinism(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from seiyuu.engines.chatterbox_engine import ChatterboxEngine

    voice_dir = tmp_path / "voice_a"
    voice_dir.mkdir()
    _write_reference(voice_dir / "reference.wav")

    eng = ChatterboxEngine(device="cuda", voices_dir=tmp_path)
    settings = {"seed": 41172, "exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8}

    first = eng.synthesize("This is a determinism check.", "voice_a", settings)
    assert first.sample_rate == 24_000 and first.duration_seconds > 0
    assert eng.conds_path("voice_a").is_file()  # conds cached

    second = eng.synthesize("This is a determinism check.", "voice_a", settings)
    # Same seed + cached conds should reproduce the segment (the frozen cache assumes this).
    assert first.samples.shape == second.samples.shape
    assert np.allclose(first.samples, second.samples, atol=1e-4), (
        "Chatterbox not reproducible with torch.manual_seed alone — the cloned-voice cache "
        "needs torch.use_deterministic_algorithms / cudnn-deterministic flags too"
    )
    eng.unload()
