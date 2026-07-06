"""Book-level smart auto-casting: hand every character a DISTINCT Kokoro voice.

The legacy ``auto_blend_recipe`` (``voices/blends.py``) hashes ``name|gender`` per character
in isolation, so two distinct characters can collide onto the same blend. ``cast_book`` sees
the whole registry at once and guarantees no two characters share a voice.

Torch-free by construction (mirrors ``voices/blends.py``): it reuses that module's ``_POOLS``
and gender->family logic and duplicates a MINIMAL preset-trait table from
``engines/kokoro_engine.py`` ``_DESCRIPTIONS`` (the same subset-duplication rationale ``_POOLS``
already uses to stay importable without loading model weights). ``test_casting`` asserts the
duplicate stays a subset of ``_POOLS`` so it can't silently drift.

Determinism + collision-freeness (the two hard guarantees):
- Characters are sorted by ``id`` (the ``blends.py`` reproducibility guarantee) so the same
  registry always yields the same voices — otherwise ``settings_hash`` would drift and force a
  silent full re-render.
- Characters are partitioned by ``(accent, family)``; each bucket draws from a DETERMINISTIC,
  DISTINCT candidate sequence (single presets first, then 2-preset blends), and each character
  consumes one candidate. Distinct candidates + one-each consumption => no collisions.
- The description/age_hint keyword bias is a TIE-BREAKER only (greedy pick among the already
  distinct remaining candidates); it can never make two characters collide.
"""

import itertools

from seiyuu.attribute.models import Character
from seiyuu.voices.blends import _POOLS, canonical_recipe

Recipe = list[tuple[str, float]]

# Family from gender, EXACTLY as auto_blend_recipe (blends.py:61-62) — unknown -> female.
_MALE = {"male", "m", "man", "boy"}

# MINIMAL torch-free trait tags DUPLICATED from kokoro_engine._DESCRIPTIONS, restricted to the
# _POOLS presets. Pure tie-breaker signal (never changes collision-freeness). Kept small on
# purpose — richer free-text reasoning is the deferred Layer-2 LLM suggester's job. A drift
# test asserts every id here lives in _POOLS.
_YOUNG = {"af_aoede", "af_sky", "am_liam", "bf_lily"}  # "youthful/younger/light" descriptions
_DEEP = {"am_adam", "am_onyx", "bm_george", "bm_daniel"}  # "deep/resonant/baritone" descriptions

# Fixed weight tiers for the blend-fallback voices. Each ordered (primary, secondary) pair at
# each weight is a UNIQUE canonical recipe, multiplying the combinatorial headroom so even a
# large single-gender cast never exhausts distinct voices.
_BLEND_WEIGHTS = (0.65, 0.6, 0.55)


def _family(gender: str | None) -> str:
    return "m" if (gender or "").strip().lower() in _MALE else "f"


def _wants(char: Character) -> set[str]:
    """Traits the character's description/age_hint asks for (lowercase keyword scan)."""
    text = f"{char.description or ''} {char.age_hint or ''}".lower()
    wants: set[str] = set()
    if any(k in text for k in ("young", "child", "boy", "girl", "teen", "kid", "little", "youth")):
        wants.add("young")
    if any(k in text for k in ("old", "elder", "deep", "gruff", "grave", "gravel", "bass")):
        wants.add("deep")
    return wants


def _voice_traits(recipe: Recipe) -> set[str]:
    """Traits of a candidate voice = traits of its PRIMARY (highest-weight) preset."""
    primary = max(recipe, key=lambda pw: pw[1])[0]
    traits: set[str] = set()
    if primary in _YOUNG:
        traits.add("young")
    if primary in _DEEP:
        traits.add("deep")
    return traits


def _candidates(pool: tuple[str, ...], reserved: set[str]):
    """Deterministic, DISTINCT voice sequence for one (accent, family) bucket.

    Distinct single presets first (skipping ``reserved`` — e.g. the narrator's preset), then
    distinct 2-preset blends (ordered pairs x weight tiers). Every yielded recipe is canonical
    and unique, so consuming one-per-character can never collide two characters.
    """
    seen: set[tuple[tuple[str, float], ...]] = set()
    for preset in pool:  # singles
        if preset in reserved:
            continue
        recipe: Recipe = [(preset, 1.0)]
        seen.add(tuple(recipe))
        yield recipe
    for weight in _BLEND_WEIGHTS:  # blend fallback: combinatorial headroom
        for primary, secondary in itertools.permutations(pool, 2):
            recipe = canonical_recipe([(primary, weight), (secondary, 1 - weight)])
            key = tuple(recipe)
            if key in seen:
                continue
            seen.add(key)
            yield recipe


def cast_book(
    characters: list[Character],
    *,
    narrator_preset: str,
    accent: str = "a",
    taken: set[str] | None = None,
) -> dict[str, Recipe]:
    """Assign each character a distinct Kokoro voice recipe (``[(preset_id, weight), ...]``).

    A 1-component recipe is a single preset; a 2-component recipe is a blend (the pool-exhaustion
    fallback). ``narrator_preset`` (and any ``taken`` presets) are excluded from single picks so
    no character shares the narrator's voice. Pure and deterministic.
    """
    reserved = set(taken or ())
    reserved.add(narrator_preset)  # no character shares the narrator's single voice

    ordered = sorted(characters, key=lambda c: c.id)  # determinism backbone
    buckets: dict[tuple[str, str], list[Character]] = {}
    for char in ordered:
        buckets.setdefault((accent, _family(char.gender)), []).append(char)

    result: dict[str, Recipe] = {}
    for key, members in buckets.items():
        pool = _POOLS[key]
        # Materialize at least the whole single-preset palette so the keyword bias has room to
        # pick a trait-matched voice, not just the first N; falls into blends only past that.
        singles_avail = sum(1 for p in pool if p not in reserved)
        count = max(len(members), singles_avail)
        pending = list(itertools.islice(_candidates(pool, reserved), count))
        if len(pending) < len(members):  # astronomically unlikely; fail loud, never collide
            raise ValueError(
                f"casting exhausted the {key} voice pool: {len(members)} characters but only "
                f"{len(pending)} distinct voices available"
            )
        # Greedy tie-breaker: each character (id order) takes the best-matching REMAINING
        # candidate; ties fall to the lowest candidate index. Bijective, so collision-free.
        for char in members:
            wants = _wants(char)
            best_i, best_score = 0, -1
            for i, cand in enumerate(pending):
                score = len(wants & _voice_traits(cand))
                if score > best_score:
                    best_score, best_i = score, i
            result[char.id] = pending.pop(best_i)
    return result
