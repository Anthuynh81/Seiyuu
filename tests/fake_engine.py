"""In-memory TTSEngine for tests: deterministic sine output, call tracking,
deliberately non-canonical native rate (8 kHz) to exercise resampling."""

import numpy as np

from seiyuu.engines.base import EngineVoice, TTSEngine


class FakeEngine(TTSEngine):
    engine_id = "fake"

    def __init__(self, fail_on: str | None = None) -> None:
        self.calls: list[tuple[str, str]] = []
        self.fail_on = fail_on

    @property
    def model_version(self) -> str:
        return "fake-1.0"

    @property
    def native_sample_rate(self) -> int:
        return 8_000

    def list_voices(self) -> list[EngineVoice]:
        return [EngineVoice(id="test_voice", name="Test Voice")]

    def cost_estimate(self, text: str) -> float:
        return 0.0

    def _synthesize_native(self, text, voice, settings):
        if self.fail_on and self.fail_on in text:
            raise RuntimeError(f"fake engine exploded on {self.fail_on!r}")
        self.calls.append((text, voice))
        seconds = max(0.05, 0.01 * len(text))
        t = np.arange(int(seconds * 8_000)) / 8_000
        return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), 8_000
