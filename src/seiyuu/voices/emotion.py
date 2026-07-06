"""Pure emotion → engine-settings mapping (F2).

A quantized :class:`EmotionVerdict` (closed label + 1..3 intensity) maps to a dict of engine
setting OVERRIDES that render merges into a segment's ``settings`` BEFORE the FROZEN
``SegmentKey`` is built — so emotion rides ``settings_hash``'s VALUE, never the key format.

Design invariants (each pinned by a fixture):
- NEUTRAL (or a missing verdict) -> ``{}``: no override, so the segment's cache key is
  byte-identical to a no-emotion render. This is what keeps ``apply_emotion=False`` and every
  neutral segment cache-stable.
- Kokoro has no emotion knob (only blend/seed/speed), so it always returns ``{}`` — injecting
  keys it ignores would churn the cache for zero audible change (a cache-stable degrade).
- Chatterbox exaggeration is CAPPED (``_MAX_EXAGGERATION``): a high-intensity tag must not
  spike the whisper-validation failure/retry rate.
- IndexTTS-2 is a stub column for M7: the SAME taxonomy will select an emotion-reference clip
  there, a pure add. Until then it returns ``{}``.

The function is pure and deterministic (no I/O, no state): identical verdicts always yield an
identical dict, so render and estimate produce identical SegmentKeys.
"""

from typing import Any

from seiyuu.attribute.models import EmotionLabel, EmotionVerdict

# Cap so a high-intensity tag can't push Chatterbox into hallucination-prone territory and
# spike whisper-validation failures. Intensity nudges within [floor, cap].
_MAX_EXAGGERATION = 0.8
_MIN_EXAGGERATION = 0.3
_INTENSITY_STEP = 0.1

# Chatterbox targets at intensity 2 (medium): (exaggeration, temperature). exaggeration drives
# emotional delivery; temperature adds a little variability for expressive states.
_CHATTERBOX_BASE: dict[EmotionLabel, tuple[float, float]] = {
    EmotionLabel.HAPPY: (0.60, 0.85),
    EmotionLabel.SAD: (0.45, 0.70),
    EmotionLabel.ANGRY: (0.75, 0.90),
    EmotionLabel.FEARFUL: (0.70, 0.90),
    EmotionLabel.TENDER: (0.45, 0.70),
    EmotionLabel.TENSE: (0.65, 0.85),
}

# ElevenLabs targets at intensity 2: (stability, style). Lower stability = more variable/
# expressive; higher style = more stylized. Intensity lowers stability and raises style.
_ELEVEN_BASE: dict[EmotionLabel, tuple[float, float]] = {
    EmotionLabel.HAPPY: (0.40, 0.45),
    EmotionLabel.SAD: (0.55, 0.20),
    EmotionLabel.ANGRY: (0.30, 0.60),
    EmotionLabel.FEARFUL: (0.30, 0.55),
    EmotionLabel.TENDER: (0.55, 0.25),
    EmotionLabel.TENSE: (0.40, 0.50),
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def map_emotion(engine: str, verdict: EmotionVerdict | None) -> dict[str, Any]:
    """Return the engine setting overrides for one emotion verdict (``{}`` = no override).

    ``engine`` is the engine id (``chatterbox``/``elevenlabs``/``kokoro``/…). A None verdict or
    a NEUTRAL label always yields ``{}`` so the segment key is unchanged. Unknown engines (and
    Kokoro, and the IndexTTS-2 M7 stub) also yield ``{}``.
    """
    if verdict is None or verdict.label is EmotionLabel.NEUTRAL:
        return {}
    step = verdict.intensity - 2  # 1..3 -> -1, 0, +1 (intensity is validated into [1, 3])
    if engine == "chatterbox":
        exaggeration, temperature = _CHATTERBOX_BASE[verdict.label]
        exaggeration = _clamp(
            exaggeration + step * _INTENSITY_STEP, _MIN_EXAGGERATION, _MAX_EXAGGERATION
        )
        temperature = _clamp(temperature + step * 0.05, 0.5, 1.0)
        return {"exaggeration": round(exaggeration, 3), "temperature": round(temperature, 3)}
    if engine == "elevenlabs":
        stability, style = _ELEVEN_BASE[verdict.label]
        stability = _clamp(stability - step * _INTENSITY_STEP, 0.1, 0.9)
        style = _clamp(style + step * _INTENSITY_STEP, 0.0, 1.0)
        return {"stability": round(stability, 3), "style": round(style, 3)}
    # kokoro: no emotion knob -> cache-stable degrade. indextts2: M7 emotion-reference column
    # (stub). Any unknown engine: no override.
    return {}
