"""Engine benchmark orchestration: resolve a voice per engine, then per engine — acquire the GPU,
time the cold load, warm up, time each segment, check seed determinism, free the GPU — and
finally (optionally) whisper-validate the collected audio on CPU.

Engines go through the ``TTSEngine`` interface (``get_engine``) and the GPU resource manager
exactly like render: ONE heavy model resident at a time. Each engine is ``unload()``-ed at the end
of its own GPU hold (so an unload failure is attributed to the right engine and stops the run),
but the card stays CLAIMED between engines: the next ``acquire`` is an in-process handoff, so the
cross-process gpu.lock is never dropped mid-run (no API server can grab the card between two
engines). The caller ``free_all()``s once at the end.

The run never loses what it measured: GPU contention, an unload failure, or Ctrl+C stop the loop
and the partial ``BenchReport`` comes back with ``aborted`` set (engines not reached: SKIPPED).

Only LOCAL engines run; paid engines are refused before construction AND by a
``cost_estimate > 0`` guard (no paid call ever happens in this automatic loop).

Validation runs AFTER the engine is unloaded so whisper's CPU load can't skew the timings.
"""

import importlib.util
import shutil
import time
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np

from seiyuu.bench.models import (
    LOCAL_ENGINES,
    AbortKind,
    BenchReport,
    BenchSegment,
    DeterminismResult,
    EngineBenchResult,
    EngineStatus,
    SegmentResult,
    VoicePlan,
    VramStats,
    finalize_engine_stats,
    rtf,
    summarize_validation,
)
from seiyuu.bench.vram import VramProbe, default_probes
from seiyuu.engines import AudioFile, TTSEngine
from seiyuu.gpu import GpuBusyError

DEFAULT_SEED = 41172  # the project's pinned default voice seed
REFERENCE_VOICE_ID = "bench_reference"

EngineFactory = Callable[[str, Path | None], TTSEngine]
ProbeFactory = Callable[[TTSEngine], list[VramProbe]]
Normalizer = Callable[[str, str], str]  # (raw text, engine_id) -> normalized text
Clock = Callable[[], float]

_SDK_MODULES = {"kokoro": "kokoro", "chatterbox": "chatterbox"}


class BenchError(Exception):
    """A whole-run problem (bad engine list, unreadable reference) — clean CLI exit."""


class BenchInterrupted(Exception):
    """Ctrl+C during an engine's measured run, raised AFTER that engine was unloaded so its
    partial ``result`` (status FAILED, reason 'interrupted ...') can still be reported."""

    def __init__(self, result: EngineBenchResult) -> None:
        super().__init__(result.reason or "interrupted")
        self.result = result


# --- voice resolution ------------------------------------------------------------------------


def parse_voice_overrides(values: list[str]) -> dict[str, str]:
    """``["kokoro=af_heart", "chatterbox=narrator_ab12"]`` -> {engine: voice_id}."""
    out: dict[str, str] = {}
    for raw in values:
        engine, sep, voice = raw.partition("=")
        engine, voice = engine.strip(), voice.strip()
        if not sep or not engine or not voice:
            raise BenchError(f"--voice expects ENGINE=VOICE_ID, got {raw!r}")
        if engine not in LOCAL_ENGINES:
            raise BenchError(f"--voice names {engine!r}; benchmarkable engines: {LOCAL_ENGINES}")
        out[engine] = voice
    return out


def check_engines(engines: list[str]) -> list[str]:
    """Dedupe (order kept) and refuse anything that isn't a local engine."""
    seen: list[str] = []
    for e in engines:
        if e not in LOCAL_ENGINES:
            raise BenchError(
                f"engine {e!r} is not benchmarkable (paid/unknown engines never run here); "
                f"choose from {', '.join(LOCAL_ENGINES)}"
            )
        if e not in seen:
            seen.append(e)
    if not seen:
        raise BenchError("no engines requested")
    return seen


def stage_reference(reference: Path, scratch_dir: Path) -> Path:
    """Copy ``reference`` into a throwaway voices dir as ``bench_reference/reference.wav``.

    NO meta.json is written: the clip never becomes a library voice and no consent record is
    persisted — the scratch dir is deleted at the end of the run (caller's job)."""
    reference = Path(reference)
    if not reference.is_file():
        raise BenchError(f"--reference not found: {reference}")
    voice_dir = Path(scratch_dir) / REFERENCE_VOICE_ID
    voice_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(reference, voice_dir / "reference.wav")
    return Path(scratch_dir)


def resolve_voice_plans(
    engines: list[str],
    *,
    library: Any,  # seiyuu.voices.VoiceLibrary
    overrides: dict[str, str] | None = None,
    reference_voices_dir: Path | None = None,
    kokoro_default: str = "af_heart",
    seed: int = DEFAULT_SEED,
    reference_clip: str | None = None,
    reference_source: str = "reference",
) -> list[VoicePlan]:
    """One ``VoicePlan`` per engine; problems become ``skip_reason`` (never an exception) so one
    unconfigured engine doesn't sink the run.

    - kokoro: ``overrides['kokoro']`` or ``kokoro_default``; a saved library preset/blend voice is
      resolved through ``render_voice_args`` (its pinned seed + recipe), a bare id is a preset.
    - chatterbox/indextts2: an explicit library voice (must be cloned + consent-verified) >
      ``--reference`` (staged in a scratch dir; ``reference_clip``/``reference_source`` record
      where it came from) > the first consent-verified cloned library voice (preferring one made
      for that engine) > skip.

    A malformed or mismatched meta.json is a skip reason (explicit voice) or is passed over (auto
    pick): one bad voice dir never sinks the run or hides the rest of the library.
    """
    from seiyuu.voices import VoiceKind, VoiceLibraryError, render_voice_args

    overrides = overrides or {}
    plans: list[VoicePlan] = []
    for engine_id in engines:
        if engine_id == "kokoro":
            vid = overrides.get("kokoro", kokoro_default)
            if library.meta_path(vid).is_file():
                try:
                    meta = library.load(vid)
                    library.verify_consent(meta)
                except (VoiceLibraryError, ValueError) as exc:  # ValueError: bad meta.json
                    plans.append(_skip(engine_id, vid, str(exc), seed))
                    continue
                if meta.engine != "kokoro":
                    plans.append(_skip(engine_id, vid, f"{vid} is a {meta.engine} voice", seed))
                    continue
                engine_voice, settings = render_voice_args(meta)
                plans.append(
                    VoicePlan(
                        engine_id=engine_id, voice_id=vid, engine_voice=engine_voice,
                        settings=settings, seed=meta.seed, source="library",
                    )
                )  # fmt: skip
            else:
                plans.append(
                    VoicePlan(engine_id=engine_id, voice_id=vid, engine_voice=vid, seed=seed)
                )
            continue

        # cloning engines
        explicit = overrides.get(engine_id)
        if explicit is not None:
            try:
                meta = library.load(explicit)
                if meta.kind is not VoiceKind.CLONED:
                    raise VoiceLibraryError(
                        f"voice {explicit!r} is a {meta.kind.value} voice; "
                        f"{engine_id} needs a cloned voice"
                    )
                library.verify_consent(meta)
            except (VoiceLibraryError, ValueError) as exc:  # ValueError: bad meta.json
                plans.append(_skip(engine_id, explicit, str(exc), seed))
                continue
            plans.append(_clone_plan(engine_id, meta, library.voices_dir, "library"))
        elif reference_voices_dir is not None:
            plans.append(
                VoicePlan(
                    engine_id=engine_id, voice_id=REFERENCE_VOICE_ID,
                    engine_voice=REFERENCE_VOICE_ID, seed=seed,
                    voices_dir=reference_voices_dir, source=reference_source,
                    reference_clip=reference_clip,
                )
            )  # fmt: skip
        else:
            meta = _auto_clone(library, engine_id)
            if meta is None:
                plans.append(
                    _skip(
                        engine_id, None,
                        f"no consent-verified cloned voice in {library.voices_dir}; pass "
                        f"--voice {engine_id}=<voice_id> or --reference <wav>",
                        seed,
                    )
                )  # fmt: skip
            else:
                plans.append(_clone_plan(engine_id, meta, library.voices_dir, "auto"))
    return plans


def _skip(engine_id: str, voice_id: str | None, reason: str, seed: int) -> VoicePlan:
    return VoicePlan(
        engine_id=engine_id, voice_id=voice_id or "-", engine_voice=voice_id or "-",
        seed=seed, skip_reason=reason,
    )  # fmt: skip


def _restarts(engine: TTSEngine) -> tuple[int, int]:
    """(worker restarts, OOMs) so far; engines without a worker process report (0, 0)."""
    return (
        int(getattr(engine, "worker_restarts", 0) or 0),
        int(getattr(engine, "worker_ooms", 0) or 0),
    )


def _clone_plan(engine_id: str, meta: Any, voices_dir: Path, source: str) -> VoicePlan:
    # A clone made for chatterbox can bench indextts2 too (both read reference.wav); only the
    # benchmarked engine's own tunables apply.
    return VoicePlan(
        engine_id=engine_id, voice_id=meta.voice_id, engine_voice=meta.voice_id,
        settings=dict(meta.settings.get(engine_id, {})), seed=meta.seed,
        voices_dir=Path(voices_dir), source=source,
    )  # fmt: skip


def _auto_clone(library: Any, engine_id: str) -> Any | None:
    """The best consent-verified local clone. Voices are loaded ONE AT A TIME so a single bad
    dir (mismatched voice_id, malformed meta.json, unreadable file) is passed over, not fatal."""
    from seiyuu.voices import VoiceKind, VoiceLibraryError

    voices_dir = Path(library.voices_dir)
    if not voices_dir.is_dir():
        return None
    candidates = []
    for d in sorted(voices_dir.iterdir()):
        if not (d / "meta.json").is_file():
            continue
        try:
            meta = library.load(d.name)
            if meta.kind is not VoiceKind.CLONED or meta.engine == "elevenlabs":
                continue
            library.verify_consent(meta)
        except (VoiceLibraryError, ValueError, OSError):  # ValueError covers pydantic errors
            continue
        # grandfathered (bool-only consent) voices skip the hash check, so make sure the audio
        # is really there — a missing reference would otherwise FAIL mid-synthesis
        if not library.reference_path(meta.voice_id).is_file():
            continue
        candidates.append(meta)
    candidates.sort(key=lambda m: (m.engine != engine_id, m.voice_id))
    return candidates[0] if candidates else None


# --- engine preflight ------------------------------------------------------------------------


def engine_preflight(engine_id: str, cfg: Any | None = None) -> str | None:
    """Why ``engine_id`` can't run on this machine (actionable), or None if it looks runnable.

    Cheap and SDK-free: a find_spec for in-process SDKs, file checks for the IndexTTS-2 worker
    (its python + checkpoints live outside this venv)."""
    if engine_id in _SDK_MODULES and importlib.util.find_spec(_SDK_MODULES[engine_id]) is None:
        return f"{engine_id}: SDK package {_SDK_MODULES[engine_id]!r} is not installed in this venv"
    if engine_id == "indextts2":
        from seiyuu.engines.indextts2_engine import checkpoints_present

        if cfg is None:
            from seiyuu.settings import get_settings

            cfg = get_settings()
        py = cfg.indextts2_worker_python
        if py is None or not Path(py).is_file():
            return (
                f"indextts2: worker python not configured or missing ({py}); set "
                "INDEXTTS2_WORKER_PYTHON in .env to the cu128 env's python.exe"
            )
        if not checkpoints_present(cfg.indextts2_checkpoints_dir):
            return (
                f"indextts2: checkpoints not found at {cfg.indextts2_checkpoints_dir}; download "
                "them and set INDEXTTS2_CHECKPOINTS_DIR"
            )
    return None


def default_engine_factory(engine_id: str, voices_dir: Path | None) -> TTSEngine:
    from seiyuu.engines import get_engine, voices_dir_kwargs

    kwargs = voices_dir_kwargs(engine_id, voices_dir) if voices_dir is not None else {}
    return get_engine(engine_id, **kwargs)


def default_normalizer(text: str, engine_id: str) -> str:
    from seiyuu.normalize import normalize_text, profile_for

    return normalize_text(text, profile=profile_for(engine_id))


# --- the measured loop -----------------------------------------------------------------------


def _determinism_index(segments: list[BenchSegment], texts: list[str]) -> int:
    """The shortest segment: cheapest to synthesize twice."""
    return min(range(len(segments)), key=lambda i: len(texts[i]))


def benchmark_engine(
    plan: VoicePlan,
    segments: list[BenchSegment],
    *,
    gpu: Any,
    engine_factory: EngineFactory = default_engine_factory,
    probe_factory: ProbeFactory | None = None,
    normalizer: Normalizer = default_normalizer,
    clock: Clock = time.perf_counter,
    warmup: int = 1,
    determinism: bool = True,
    progress: Callable[[str], None] | None = None,
    preflight: Callable[[str], str | None] | None = None,
) -> tuple[EngineBenchResult, list[AudioFile]]:
    """Benchmark one engine. Returns the result + the timed segments' audio (for validation /
    saving). An engine problem, including its own ``unload()`` failing, becomes SKIPPED/FAILED
    with a reason (``unload_failed`` set for the latter), never an exception. Only two things
    escape: ``GpuBusyError`` (another seiyuu process holds the card; nothing was loaded) and
    ``BenchInterrupted`` (Ctrl+C; raised after the engine was unloaded, carrying the partial
    result).

    The engine is unloaded at the end of its own GPU hold, but the card is NOT released (no
    ``free_all``): the manager keeps it claimed and the next ``acquire`` hands off in-process."""
    say = progress or (lambda _m: None)
    result = EngineBenchResult(
        engine=plan.engine_id, voice_id=plan.voice_id, voice_source=plan.source, seed=plan.seed,
        reference_clip=plan.reference_clip,
    )  # fmt: skip
    audio: list[AudioFile] = []

    def skip(reason: str) -> tuple[EngineBenchResult, list[AudioFile]]:
        result.status, result.reason = EngineStatus.SKIPPED, reason
        say(f"[{plan.engine_id}] skipped: {reason}")
        return result, audio

    if plan.engine_id not in LOCAL_ENGINES:
        return skip(f"{plan.engine_id} is not a local engine; paid engines never benchmark")
    if plan.skip_reason:
        return skip(plan.skip_reason)
    if not segments:
        return skip("no segments to synthesize")
    if preflight is not None and (why := preflight(plan.engine_id)):
        return skip(why)
    try:
        engine = engine_factory(plan.engine_id, plan.voices_dir)
    except Exception as exc:  # construction is SDK-free; a failure here is config, not a bug
        return skip(f"cannot construct engine: {exc}")
    texts = [normalizer(s.text, plan.engine_id) for s in segments]
    if any(engine.cost_estimate(t) > 0 for t in texts):
        return skip(f"{plan.engine_id} reports a non-zero cost; paid engines never benchmark")
    try:
        result.model_version = engine.model_version
    except Exception as exc:  # e.g. indextts2 fingerprint with no checkpoints
        return skip(f"model_version unavailable: {exc}")
    result.uses_gpu = engine.uses_gpu

    probes = (probe_factory or default_probes)(engine) if engine.uses_gpu else []
    vram = VramStats()
    synth_settings = {**plan.settings, "seed": plan.seed}
    restarts0, ooms0 = _restarts(engine)
    stage = "acquire GPU"
    unload_error: str | None = None
    interrupted = False

    def synth(text: str) -> AudioFile:
        return engine.synthesize(text, plan.engine_voice, synth_settings)

    try:
        with gpu.acquire(engine, engine.engine_id) if engine.uses_gpu else nullcontext():
            try:
                for p in probes:
                    p.begin()
                try:
                    stage = "cold load (warm)"
                    say(f"[{plan.engine_id}] loading model ...")
                    t0 = clock()
                    engine.warm()
                    result.cold_load_s = clock() - t0
                    for p in probes:
                        p.mark_loaded()
                    say(f"[{plan.engine_id}] loaded in {result.cold_load_s:.1f}s")

                    if warmup > 0:
                        stage = "warm-up synthesis"
                        t0 = clock()
                        for w in range(warmup):
                            synth(texts[w % len(texts)])
                        result.first_synth_s = clock() - t0

                    for i, (seg, text) in enumerate(zip(segments, texts, strict=True)):
                        stage = f"segment {seg.index}"
                        before = _restarts(engine)[0]
                        t0 = clock()
                        clip = synth(text)
                        wall = clock() - t0
                        restarted = _restarts(engine)[0] - before
                        audio.append(clip)
                        result.segments.append(
                            SegmentResult(
                                index=seg.index, kind=seg.kind, chars=len(text),
                                words=len(text.split()), wall_s=wall,
                                audio_s=clip.duration_seconds,
                                rtf=rtf(wall, clip.duration_seconds),
                                worker_restarts=restarted,
                            )
                        )  # fmt: skip
                        r = result.segments[-1].rtf
                        say(
                            f"[{plan.engine_id}] {i + 1}/{len(segments)} {seg.kind.value}: "
                            f"{clip.duration_seconds:.1f}s audio in {wall:.2f}s"
                            + (f" (RTF {r:.2f})" if r is not None else "")
                            + (f"  WORKER RESTARTED x{restarted} (OOM/crash)" if restarted else "")
                        )

                    if determinism:
                        stage = "determinism check"
                        di = _determinism_index(segments, texts)
                        a = synth(texts[di]).samples
                        b = synth(texts[di]).samples
                        result.determinism = compare_samples(a, b, segments[di].index, plan.seed)
                finally:
                    for p in probes:
                        p.end(vram)  # measured BEFORE the unload below
            finally:
                # Unload inside the hold: the manager still lists this engine as resident (the
                # next acquire's handoff unload is an idempotent no-op) and the card stays
                # claimed. Guarded so an unload error is recorded on THIS engine instead of being
                # raised out of a finally over the real failure.
                try:
                    engine.unload()
                except Exception as exc:
                    unload_error = f"{type(exc).__name__}: {exc}"
    except GpuBusyError:
        raise
    except KeyboardInterrupt:
        interrupted = True
        result.status = EngineStatus.FAILED
        result.reason = f"interrupted (Ctrl+C) during {stage}"
        say(f"[{plan.engine_id}] {result.reason}")
    except Exception as exc:
        result.status = EngineStatus.FAILED
        if stage == "acquire GPU":
            # acquire's only non-busy failure is the PREVIOUS resident's handoff unload: the card
            # may still hold that model, so nothing more may load on top of it
            result.unload_failed = True
            result.reason = (
                f"{plan.engine_id} not loaded: the GPU handoff (unloading the previous engine) "
                f"failed: {type(exc).__name__}: {exc}"
            )
        else:
            result.reason = f"{plan.engine_id} failed during {stage}: {type(exc).__name__}: {exc}"
        say(f"[{plan.engine_id}] FAILED: {result.reason}")
    if unload_error is not None:
        result.status = EngineStatus.FAILED
        result.unload_failed = True
        _note(result, f"unload failed: {unload_error}")
        say(f"[{plan.engine_id}] FAILED: unload failed: {unload_error}")
    restarts1, ooms1 = _restarts(engine)
    result.worker_restarts, result.worker_ooms = restarts1 - restarts0, ooms1 - ooms0
    if result.worker_restarts:
        say(
            f"[{plan.engine_id}] WARNING: worker restarted {result.worker_restarts}x "
            f"({result.worker_ooms} OOM); those timings include a full model reload"
        )
    result.vram = vram if vram.methods else None
    finalize_engine_stats(result)
    if interrupted:
        raise BenchInterrupted(result)
    return result, audio


def compare_samples(a: np.ndarray, b: np.ndarray, index: int, seed: int) -> DeterminismResult:
    same_len = a.shape == b.shape
    identical = bool(same_len and np.array_equal(a, b))
    diff = float(np.max(np.abs(a - b))) if same_len and a.size else (0.0 if same_len else None)
    return DeterminismResult(
        segment_index=index, seed=seed, identical=identical, same_length=same_len, max_abs_diff=diff
    )


def validate_engine_audio(
    result: EngineBenchResult,
    audio: list[AudioFile],
    texts: list[str],
    validator: Any,
) -> None:
    """Whisper-score each timed segment (CPU) into ``result`` (mutates). Uses the same validator
    the render loop uses, fed the in-memory 16 kHz waveform."""
    from seiyuu.validate import resample_to_whisper

    for seg, clip, text in zip(result.segments, audio, texts, strict=False):
        verdict = validator.validate(resample_to_whisper(clip.samples, clip.sample_rate), text)
        seg.validation_score = verdict.score
        seg.validation_ok = verdict.ok
        seg.transcript = verdict.transcript
    result.validation = summarize_validation(result.segments, float(validator.min_ratio))


def run_benchmark(
    plans: list[VoicePlan],
    segments: list[BenchSegment],
    *,
    gpu: Any,
    segments_source: str = "bundled",
    engine_factory: EngineFactory = default_engine_factory,
    probe_factory: ProbeFactory | None = None,
    normalizer: Normalizer = default_normalizer,
    clock: Clock = time.perf_counter,
    warmup: int = 1,
    determinism: bool = True,
    validator: Any | None = None,
    audio_dir: Path | None = None,
    progress: Callable[[str], None] | None = None,
    preflight: Callable[[str], str | None] | None = engine_preflight,
    environment: dict[str, Any] | None = None,
) -> BenchReport:
    """Benchmark every plan in order (one engine resident at a time) into a ``BenchReport``.

    Never raises for GPU contention, an unload failure, or Ctrl+C: the loop stops, ``aborted`` /
    ``abort_detail`` say why, every engine already measured stays in ``results``, and engines not
    reached are SKIPPED, so the caller can always write the report."""
    say = progress or (lambda _m: None)
    report = BenchReport(
        segments_source=segments_source,
        segment_count=len(segments),
        warmup=warmup,
        environment=environment or {},
    )

    def abort(kind: AbortKind, detail: str) -> None:
        report.aborted, report.abort_detail = kind, detail
        say(f"run stopped ({kind.value}): {detail}")

    remaining = list(plans)
    while remaining:
        plan = remaining.pop(0)
        try:
            result, audio = benchmark_engine(
                plan, segments, gpu=gpu, engine_factory=engine_factory,
                probe_factory=probe_factory, normalizer=normalizer, clock=clock, warmup=warmup,
                determinism=determinism, progress=progress, preflight=preflight,
            )  # fmt: skip
        except GpuBusyError as exc:
            report.results.append(_not_run(plan, f"GPU busy: {exc}"))
            abort(AbortKind.GPU_BUSY, str(exc))
            break
        except BenchInterrupted as exc:
            report.results.append(exc.result)
            abort(AbortKind.INTERRUPTED, exc.result.reason or "interrupted (Ctrl+C)")
            break
        except KeyboardInterrupt:  # outside the measured part (construction, normalization)
            report.results.append(_not_run(plan, "interrupted (Ctrl+C) before it started"))
            abort(AbortKind.INTERRUPTED, "interrupted (Ctrl+C)")
            break
        try:
            if audio and audio_dir is not None:
                engine_dir = Path(audio_dir) / plan.engine_id
                for seg, clip in zip(result.segments, audio, strict=False):
                    clip.save(engine_dir / f"{seg.index:02d}_{seg.kind.value}.wav")
            if audio and validator is not None:
                texts = [normalizer(s.text, plan.engine_id) for s in segments]
                say(f"[{plan.engine_id}] whisper-validating {len(audio)} segment(s) on CPU")
                try:
                    validate_engine_audio(result, audio, texts, validator)
                except Exception as exc:  # validation is optional: keep the timings
                    _note(result, f"validation failed: {type(exc).__name__}: {exc}")
        except KeyboardInterrupt:
            _note(result, "interrupted (Ctrl+C) while saving/validating; the timings are complete")
            report.results.append(result)
            abort(AbortKind.INTERRUPTED, f"interrupted (Ctrl+C) after timing {plan.engine_id}")
            break
        report.results.append(result)
        if result.unload_failed:
            # one heavy model at a time: never load another on top of one that may still be there
            abort(AbortKind.UNLOAD_FAILED, f"{plan.engine_id}: {result.reason}")
            break
    for plan in remaining:
        report.results.append(_not_run(plan, f"not run: the run stopped ({report.abort_detail})"))
    return report


def _note(result: EngineBenchResult, note: str) -> None:
    result.reason = f"{result.reason}; {note}" if result.reason else note


def _not_run(plan: VoicePlan, reason: str) -> EngineBenchResult:
    return EngineBenchResult(
        engine=plan.engine_id, voice_id=plan.voice_id, voice_source=plan.source, seed=plan.seed,
        reference_clip=plan.reference_clip, status=EngineStatus.SKIPPED, reason=reason,
    )  # fmt: skip
