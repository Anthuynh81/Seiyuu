"""Section 2 demo: list Kokoro voices, synthesize one sentence, save a WAV.

Run:  uv run python scripts/demo_section2.py [voice_id]
"""

import sys
from pathlib import Path

from seiyuu.engines import get_engine
from seiyuu.settings import get_settings

SENTENCE = (
    "It is a truth universally acknowledged, that a single man in possession "
    "of a good fortune must be in want of a wife."
)

engine = get_engine("kokoro")

print(f"engine: {engine.engine_id}  model: {engine.model_version}\n")
print("available voices:")
for v in engine.list_voices():
    print(f"  {v.id:<14} {v.language}  {v.gender}")

voice = sys.argv[1] if len(sys.argv) > 1 else get_settings().kokoro_default_voice
print(f"\nsynthesizing with {voice!r} ...")
audio = engine.synthesize(SENTENCE, voice, settings={"seed": 41172})

out = Path(get_settings().output_dir) / f"section2_demo_{voice}.wav"
audio.save(out)
print(f"{audio.duration_seconds:.1f}s of audio -> {out}")
