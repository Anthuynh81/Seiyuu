"""Deterministic Kokoro blend recipes (pure; the tensor math lives in KokoroEngine).

A blend recipe is a canonical, sorted list of (preset_id, weight) with weights normalized to
sum 1 and rounded. Folding that canonical form into the render settings keeps settings_hash
stable, so the same intended voice always hits the same cache entry. Auto-draft blends are
derived deterministically from a character's name+gender, so re-running `assign` reproduces
the same draft voices.
"""

import hashlib

from seiyuu.voices.models import BlendComponent, VoiceKind, VoiceMeta

_WEIGHT_PRECISION = 4

# Curated American/British female/male Kokoro v1.0 presets for auto-draft blends. Kept here
# (a subset of KokoroEngine._PRESETS) so this module stays torch-free.
_POOLS: dict[tuple[str, str], tuple[str, ...]] = {
    ("a", "f"): ("af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky", "af_aoede"),
    ("a", "m"): ("am_adam", "am_michael", "am_eric", "am_liam", "am_onyx", "am_puck"),
    ("b", "f"): ("bf_emma", "bf_alice", "bf_isabella", "bf_lily"),
    ("b", "m"): ("bm_george", "bm_lewis", "bm_daniel", "bm_fable"),
}


def canonical_recipe(components) -> list[tuple[str, float]]:
    """Normalize weights to sum 1, round, and sort by preset_id — the cache-stable form."""
    pairs = [
        (c.preset_id, c.weight) if isinstance(c, BlendComponent) else (str(c[0]), float(c[1]))
        for c in components
    ]
    total = sum(w for _, w in pairs) or 1.0
    return sorted(
        ((p, round(w / total, _WEIGHT_PRECISION)) for p, w in pairs), key=lambda pw: pw[0]
    )


def render_voice_args(meta: VoiceMeta) -> tuple[str, dict]:
    """The (engine voice arg, settings) an adapter needs to synthesize `meta`.

    The FROZEN SegmentKey still keys on ``meta.voice_id`` — this is ONLY the adapter-facing
    pair. A Kokoro preset is addressed by its ``preset_id`` (not the library voice_id, which
    the engine doesn't know); a blend folds its canonical recipe into settings (and the engine
    builds the weighted voicepack from it); a cloned voice is addressed by voice_id (the engine
    resolves its conds cache from that). The returned settings are used for BOTH the cache key
    and the synth call, so the recipe stays part of the key for blends.
    """
    settings = meta.engine_settings()
    if meta.kind is VoiceKind.PRESET and meta.preset_id:
        return meta.preset_id, settings
    if meta.kind is VoiceKind.BLEND and meta.blend:
        recipe = [list(pw) for pw in canonical_recipe(meta.blend)]
        return meta.voice_id, {**settings, "blend": recipe}
    return meta.voice_id, settings


def auto_blend_recipe(
    name: str, gender: str | None, *, accent: str = "a"
) -> list[tuple[str, float]]:
    """A deterministic 2-preset same-family draft blend for a character."""
    gl = (gender or "").strip().lower()
    family = "m" if gl in {"male", "m", "man", "boy"} else "f"  # default female
    pool = _POOLS[(accent, family)]
    h = int(hashlib.sha256(f"{name}|{gender}".encode()).hexdigest(), 16)
    first = h % len(pool)
    second = (first + 1 + (h // len(pool)) % (len(pool) - 1)) % len(pool)
    primary = 0.5 + ((h >> 16) % 26) / 100  # 0.50..0.75, deterministic
    return canonical_recipe([(pool[first], primary), (pool[second], 1 - primary)])
