"""Inter-voice loudness matching: one gain per voice per book, applied before concatenation.

Different engines (and different cloned references) land at very different levels: a quiet
cloned character next to a loud Kokoro narrator keeps a volume jump at every speaker change,
and the whole-chapter loudnorm that runs afterwards cannot fix a jump *inside* the chapter.

Method:
- Each segment's ITU-R BS.1770 integrated loudness is measured with pyloudnorm. Segments
  shorter than one 0.4 s gating block, or that measure -inf (silence), are skipped.
- A voice's loudness is the duration-weighted power mean of its segments:
  ``L = 10*log10(sum(d_i * 10^(L_i/10)) / sum(d_i))``.
- Target: the narrator's loudness when the manifest's assignment names one (narration
  dominates a book, so the narrator stays untouched and characters move to it); otherwise
  the duration-weighted median across voices.
- Gain per voice = target - L_voice, clamped to +/- ``max_gain_db`` so a broken measurement
  cannot blow a voice up, then peak-guarded: a boost is reduced (never hard-clipped) so no
  sample of that voice goes past ``peak_ceiling_dbfs``.

ONE gain per voice per book keeps each voice's own dynamics (a whispered line stays quieter
than a shouted one). Single-voice renders are a no-op. Cached segment WAVs are never touched:
gains are applied in memory. Per-segment measurements are cached in ``loudness_cache.json``
beside the manifest (keyed by wav path + size + mtime_ns) so assemble then master measures
once; a corrupt or missing cache just means re-measuring.
"""

import json
import math
import os
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from seiyuu.assemble.errors import AssembleError
from seiyuu.ingest.models import BlockType
from seiyuu.render.models import RenderManifest
from seiyuu.repository.atomic import atomic_write_text

LOUDNESS_CACHE_NAME = "loudness_cache.json"
_CACHE_VERSION = 1
MAX_VOICE_GAIN_DB = 9.0
PEAK_CEILING_DBFS = -1.0
_BLOCK_SECONDS = 0.4  # BS.1770 gating block; pyloudnorm refuses anything shorter
_CANCEL_EVERY = 200  # segments between cancel checks
_PROGRESS_EVERY = 1000  # freshly measured segments between progress lines


@dataclass(frozen=True)
class VoiceMatch:
    """Inter-voice loudness matching knobs (None at the call site = matching disabled)."""

    max_gain_db: float = MAX_VOICE_GAIN_DB
    peak_ceiling_dbfs: float = PEAK_CEILING_DBFS

    def __post_init__(self) -> None:
        # A negative bound would invert the +/- clamp and boost every voice (narrator too).
        if not math.isfinite(self.max_gain_db) or self.max_gain_db < 0:
            raise ValueError(f"max_gain_db must be a finite value >= 0, got {self.max_gain_db}")


@dataclass(frozen=True)
class SegmentLevel:
    """One segment's measurement. ``lufs`` is None when unmeasurable (too short / silent)."""

    lufs: float | None
    seconds: float
    peak: float  # linear sample peak (abs max), for the peak guard


@dataclass(frozen=True)
class VoiceGain:
    voice_id: str
    loudness: float  # measured duration-weighted LUFS
    wanted_db: float  # target - loudness, before clamp/peak guard
    gain_db: float  # what is actually applied
    clamped: bool = False
    peak_limited: bool = False

    @property
    def factor(self) -> float:
        return 10.0 ** (self.gain_db / 20.0)


@dataclass(frozen=True)
class VoiceGainPlan:
    target_lufs: float
    target_source: str  # "narrator <id>" or "median"
    gains: dict[str, VoiceGain]

    def factors(self) -> dict[str, float]:
        """voice_id -> linear factor, omitting exact no-ops (keeps those samples bit-exact)."""
        return {v: g.factor for v, g in self.gains.items() if g.gain_db != 0.0}

    def summary(self) -> str:
        values = [g.gain_db for g in self.gains.values()]
        line = (
            f"matching voice loudness: {len(self.gains)} voices, "
            f"gains {min(values):+.1f}..{max(values):+.1f} dB "
            f"(target {self.target_lufs:.1f} LUFS, {self.target_source})"
        )
        clamped = [g for g in self.gains.values() if g.clamped]
        limited = [g for g in self.gains.values() if g.peak_limited]
        if clamped:
            line += "; clamped: " + ", ".join(
                f"{g.voice_id} ({g.wanted_db:+.1f} -> {g.gain_db:+.1f} dB)" for g in clamped
            )
        if limited:
            line += "; peak-limited: " + ", ".join(
                f"{g.voice_id} ({g.gain_db:+.1f} dB)" for g in limited
            )
        return line


def measure_segment(wav_path: Path) -> SegmentLevel:
    """BS.1770 integrated loudness + sample peak of one canonical segment WAV."""
    import pyloudnorm

    samples, rate = sf.read(str(wav_path), dtype="float32")
    if samples.ndim != 1:
        samples = samples.mean(axis=1)
    seconds = len(samples) / rate if rate else 0.0
    peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
    lufs: float | None = None
    if seconds >= _BLOCK_SECONDS and peak > 0.0:
        with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
            warnings.simplefilter("ignore")
            value = float(pyloudnorm.Meter(rate).integrated_loudness(samples.astype(np.float64)))
        if math.isfinite(value):
            lufs = value
    return SegmentLevel(lufs=lufs, seconds=seconds, peak=peak)


def _load_cache(path: Path) -> dict[str, dict]:
    """Corrupt, foreign or missing cache == empty (re-measure), never an error."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") != _CACHE_VERSION or not isinstance(data.get("segments"), dict):
            return {}
        return data["segments"]
    except (OSError, ValueError, AttributeError):
        return {}


def _cached_level(entry: object, size: int, mtime_ns: int) -> SegmentLevel | None:
    if not isinstance(entry, dict):
        return None
    if entry.get("size") != size or entry.get("mtime_ns") != mtime_ns:
        return None
    try:
        lufs = entry["lufs"]
        return SegmentLevel(
            lufs=None if lufs is None else float(lufs),
            seconds=float(entry["seconds"]),
            peak=float(entry["peak"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _power_mean(levels: list[SegmentLevel]) -> float | None:
    measured = [(lv.seconds, lv.lufs) for lv in levels if lv.lufs is not None]
    total = sum(d for d, _ in measured)
    if not measured or total <= 0:
        return None
    return 10.0 * math.log10(sum(d * 10.0 ** (lufs / 10.0) for d, lufs in measured) / total)


def _weighted_median(values: list[tuple[float, float]]) -> float:
    """values = (loudness, weight); the loudness where cumulative weight crosses half."""
    ordered = sorted(values)
    half = sum(w for _, w in ordered) / 2.0
    acc = 0.0
    for value, weight in ordered:
        acc += weight
        if acc >= half:
            return value
    return ordered[-1][0]


def plan_voice_gains(
    manifest: RenderManifest,
    book_dir: Path,
    match: VoiceMatch,
    *,
    measure: Callable[[Path], SegmentLevel] | None = None,
    check_cancel: Callable[[], None] | None = None,
    progress: Callable[[str], None] | None = None,
) -> VoiceGainPlan | None:
    """Measure every voiced segment (cache-backed) and plan one gain per voice.

    Returns None when there is nothing to match (fewer than two voices with measurable
    audio). Missing segment WAVs are skipped here; `_chapter_samples` reports them loudly.
    An unreadable segment WAV raises AssembleError naming its chapter and block.
    """
    measure = measure or measure_segment
    check = check_cancel or (lambda: None)
    say = progress or (lambda _msg: None)
    # voice_id -> [(wav rel path, chapter index, block id)]
    by_voice: dict[str, list[tuple[str, int, str]]] = {}
    for chapter in manifest.chapters:
        for seg in chapter.segments:
            if seg.type is BlockType.SCENE_BREAK or seg.wav is None or seg.voice_id is None:
                continue
            by_voice.setdefault(seg.voice_id, []).append((seg.wav, chapter.index, seg.block_id))
    if len(by_voice) < 2:
        return None  # single-voice: no-op, nothing measured

    cache_path = book_dir / LOUDNESS_CACHE_NAME
    old = _load_cache(cache_path)
    fresh: dict[str, dict] = {}
    levels: dict[str, list[SegmentLevel]] = {}

    def record(voice_id: str, rel: str, st: os.stat_result, level: SegmentLevel) -> None:
        fresh[rel] = {
            "size": st.st_size,
            "mtime_ns": st.st_mtime_ns,
            "lufs": level.lufs,
            "seconds": level.seconds,
            "peak": level.peak,
        }
        levels.setdefault(voice_id, []).append(level)

    try:
        # Pass 1 (cheap: stat + cache lookup) splits cached from to-be-measured segments,
        # so the slow pass can announce itself up front.
        pending: list[tuple[str, str, int, str, os.stat_result]] = []
        seen = 0
        for voice_id, wavs in by_voice.items():
            for rel, chapter_index, block_id in wavs:
                if seen % _CANCEL_EVERY == 0:
                    check()
                seen += 1
                try:
                    st = (book_dir / rel).stat()
                except OSError:
                    continue
                level = _cached_level(old.get(rel), st.st_size, st.st_mtime_ns)
                if level is None:
                    pending.append((voice_id, rel, chapter_index, block_id, st))
                else:
                    record(voice_id, rel, st, level)

        # Pass 2: measure what the cache could not answer (minutes on a long first pass).
        if pending:
            say(
                f"matching voice loudness: measuring {len(pending)} segments "
                f"({len(fresh)} cached)..."
            )
        for done, (voice_id, rel, chapter_index, block_id, st) in enumerate(pending):
            if done % _CANCEL_EVERY == 0:
                check()  # a long first measurement pass stays cancellable
            if done and done % _PROGRESS_EVERY == 0:
                say(f"matching voice loudness: measured {done}/{len(pending)} segments")
            path = book_dir / rel
            try:
                level = measure(path)
            except (OSError, RuntimeError, ValueError) as exc:  # LibsndfileError is a RuntimeError
                raise AssembleError(
                    f"unreadable segment audio: chapter={chapter_index} block={block_id} "
                    f"{path}: {exc}; re-run `seiyuu render`"
                ) from exc
            record(voice_id, rel, st, level)
    finally:
        # Persist even a cancelled/failed pass: valid entries not reached yet are kept, fresh
        # ones are added, and anything outside this manifest is pruned (bounded cache).
        keep = {rel for wavs in by_voice.values() for rel, _, _ in wavs}
        merged = {k: v for k, v in old.items() if k in keep and k not in fresh}
        merged.update(fresh)
        if merged != old:
            payload = json.dumps({"version": _CACHE_VERSION, "segments": merged}, indent=1)
            try:
                atomic_write_text(cache_path, payload)
            except OSError:
                pass  # a cache is an optimization; never fail assembly over it

    loudness: dict[str, float] = {}
    weight: dict[str, float] = {}
    peaks: dict[str, float] = {}
    for voice_id, voice_levels in levels.items():
        value = _power_mean(voice_levels)
        if value is None:
            continue  # no measurable segment: left untouched
        loudness[voice_id] = value
        weight[voice_id] = sum(lv.seconds for lv in voice_levels if lv.lufs is not None)
        peaks[voice_id] = max(lv.peak for lv in voice_levels)
    if len(loudness) < 2:
        return None

    narrator = (manifest.assignment or {}).get("narrator_voice_id")
    if narrator in loudness:
        target, source = loudness[narrator], f"narrator {narrator}"
    else:
        target = _weighted_median([(loudness[v], weight[v]) for v in loudness])
        source = "median"

    ceiling = 10.0 ** (match.peak_ceiling_dbfs / 20.0)
    gains: dict[str, VoiceGain] = {}
    for voice_id, value in loudness.items():
        wanted = target - value
        gain = max(-match.max_gain_db, min(match.max_gain_db, wanted))
        clamped = gain != wanted
        peak_limited = False
        if gain > 0.0 and peaks[voice_id] > 0.0:
            # Reduce the boost rather than hard-clip: the loudest sample lands at the ceiling.
            headroom = max(0.0, 20.0 * math.log10(ceiling / peaks[voice_id]))
            if gain > headroom:
                gain, peak_limited = headroom, True
        gains[voice_id] = VoiceGain(
            voice_id=voice_id,
            loudness=value,
            wanted_db=wanted,
            gain_db=gain,
            clamped=clamped,
            peak_limited=peak_limited,
        )
    return VoiceGainPlan(target_lufs=target, target_source=source, gains=gains)


__all__ = [
    "LOUDNESS_CACHE_NAME",
    "MAX_VOICE_GAIN_DB",
    "PEAK_CEILING_DBFS",
    "SegmentLevel",
    "VoiceGain",
    "VoiceGainPlan",
    "VoiceMatch",
    "measure_segment",
    "plan_voice_gains",
]
