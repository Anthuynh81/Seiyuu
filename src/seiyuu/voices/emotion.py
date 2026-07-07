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
- IndexTTS-2 (M7) maps the SAME taxonomy to its native 8-dim ``emo_vector`` (one weighted
  dominant dimension) plus an ``emo_alpha`` strength scalar carrying intensity — a pure add
  that rides ``settings_hash``'s VALUE like the other columns. NEUTRAL still returns ``{}`` so
  the engine takes its neutral-from-speaker-clip path (a byte-stable, un-emotive render).

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


# IndexTTS-2 emotion vector: 8 weights in a FIXED upstream order (infer_v2.py):
#   [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
# We drive ONE dominant dimension per label (direction) and carry intensity in emo_alpha
# (IndexTTS-2's native 0..1 strength scalar), NOT by inflating the vector. TENSE -> afraid is a
# SINGLE dim on purpose: summing two high-arousal dims (afraid+angry) is perceptually unvalidated
# and compounds unpredictably under alpha scaling — keep it one dim until audition tuning (M7-1).
# disgusted/surprised/melancholic are left at 0 (harmless; unused dims don't distort the others).
# The magnitudes below are conservative INITIAL values, tunable by audition after the M7 spike;
# changing them churns settings_hash for emotive segments only (NEUTRAL stays byte-stable).
_INDEXTTS2_DIM: dict[EmotionLabel, int] = {
    EmotionLabel.HAPPY: 0,  # happy
    EmotionLabel.ANGRY: 1,  # angry
    EmotionLabel.SAD: 2,  # sad
    EmotionLabel.FEARFUL: 3,  # afraid
    EmotionLabel.TENSE: 3,  # afraid (closest single dim)
    EmotionLabel.TENDER: 7,  # calm
}
# Direction weight on the dominant dim — capped well below 1.0 so a high-intensity tag can't
# overdrive the autoregressive model into hallucination (same discipline as _MAX_EXAGGERATION).
_INDEXTTS2_WEIGHT = 0.8
# Intensity 1/2/3 -> emo_alpha (low/medium/high). Bounded to <= 1.0 (the upstream max).
_INDEXTTS2_ALPHA: dict[int, float] = {1: 0.6, 2: 0.8, 3: 1.0}


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
    if engine == "indextts2":
        dim = _INDEXTTS2_DIM.get(verdict.label)
        if dim is None:  # a label with no clean dim degrades to neutral (byte-stable)
            return {}
        vector = [0.0] * 8
        vector[dim] = _INDEXTTS2_WEIGHT
        alpha = _INDEXTTS2_ALPHA[verdict.intensity]
        return {"emo_vector": [round(x, 3) for x in vector], "emo_alpha": round(alpha, 3)}
    # kokoro: no emotion knob -> cache-stable degrade. Any unknown engine: no override.
    return {}
