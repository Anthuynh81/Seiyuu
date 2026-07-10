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
# and compounds unpredictably under alpha scaling.
# disgusted/surprised/melancholic are left at 0 (harmless; unused dims don't distort the others).
#
# Per-label (dominant dim, direction weight, MAX alpha), TUNED BY AUDITION (2026-07-10,
# output/emotion_audition — full matrix + angry weight×alpha sweep on the reference voice):
# - angry reads best as a full-strength direction with moderated alpha (w1.0×a0.8 beat
#   w0.8×a1.0 in the sweep), so its vector is 1.0 and its ladder tops at 0.8;
# - sad and tense OVERACT at alpha 1.0 (their intensity-3 clips degraded), so their ladders
#   top at 0.8 — intensity 3 lands on the approved intensity-2 sound;
# - happy/fearful/tender keep the initial values (their intensity-3 clips auditioned well).
# Changing these churns settings_hash for that label's emotive segments only (NEUTRAL and
# untouched labels stay byte-stable — happy/fearful/tender are unchanged from the initial values).
_INDEXTTS2_MAP: dict[EmotionLabel, tuple[int, float, float]] = {
    EmotionLabel.HAPPY: (0, 0.8, 1.0),
    EmotionLabel.ANGRY: (1, 1.0, 0.8),
    EmotionLabel.SAD: (2, 0.8, 0.8),
    EmotionLabel.FEARFUL: (3, 0.8, 1.0),
    EmotionLabel.TENSE: (3, 0.8, 0.8),  # afraid dim; full alpha reads as panic, not tension
    EmotionLabel.TENDER: (7, 0.8, 1.0),  # calm
}
# Intensity 1/2/3 -> this fraction of the label's max alpha (low/medium/high).
_INDEXTTS2_ALPHA_RATIO: dict[int, float] = {1: 0.6, 2: 0.8, 3: 1.0}


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
        entry = _INDEXTTS2_MAP.get(verdict.label)
        if entry is None:  # a label with no clean dim degrades to neutral (byte-stable)
            return {}
        dim, weight, max_alpha = entry
        vector = [0.0] * 8
        vector[dim] = weight
        alpha = _INDEXTTS2_ALPHA_RATIO[verdict.intensity] * max_alpha
        return {"emo_vector": [round(x, 3) for x in vector], "emo_alpha": round(alpha, 3)}
    # kokoro: no emotion knob -> cache-stable degrade. Any unknown engine: no override.
    return {}
