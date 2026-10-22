"""Benchmark the LOCAL TTS engines (kokoro / chatterbox / indextts2) on this machine's GPU.

Re-measures what the old 8GB-card notes recorded: cold load time, warm per-segment wall time,
RTF (wall / audio), peak VRAM, same-seed determinism, and (opt-in) whisper pass rate. Engines run
one at a time through the TTSEngine interface + the GPU resource manager (one heavy model
resident; each engine is freed before the next loads). Paid engines are never run.

Voices: kokoro uses --voice kokoro=<preset or library id> (default: settings.kokoro_default_voice).
Cloning engines use --voice <engine>=<library voice id>, else --reference <wav>, else the first
consent-verified cloned voice in the library, else (only with --use-test-reference) the configured
test clip (settings.indextts2_test_reference). An ad-hoc clip (--reference / --use-test-reference)
is cloned only with --consent, the same attestation `seiyuu voices clone` demands; it is staged in
a throwaway voices dir (no meta.json, consent NOT persisted, nothing added to your library) and
recorded as an ad-hoc clip in bench.json. An engine that isn't set up (e.g. indextts2 worker python
/ checkpoints missing) is SKIPPED with the reason, not a traceback.

Output: <out>/bench.json + bench.md (+ <out>/audio/<engine>/*.wav unless --no-audio). The report is
written even when the run stops early (GPU busy, unload failure, Ctrl+C).

Exit codes: 0 = done (at least one engine measured, none failed); 1 = an engine FAILED;
2 = usage error / GPU busy; 3 = nothing was benchmarked (every engine skipped); 130 = Ctrl+C.

Usage (GPU job — loads each engine's weights; IndexTTS-2 is the slow one):
    uv run python scripts/benchmark_engines.py                          # all 3, bundled texts
    uv run python scripts/benchmark_engines.py --engines kokoro chatterbox --validate
    uv run python scripts/benchmark_engines.py --voice chatterbox=narrator_ab12
    uv run python scripts/benchmark_engines.py --reference clip.wav --consent --engines indextts2
    uv run python scripts/benchmark_engines.py --use-test-reference --consent
    uv run python scripts/benchmark_engines.py --book pride --chapter 3 --limit 8
"""

import argparse
import sys
import tempfile
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from seiyuu.bench import (  # noqa: E402
    LOCAL_ENGINES,
    AbortKind,
    BenchError,
    BenchInputError,
    BenchReport,
    EngineStatus,
    book_segments,
    bundled_segments,
    console_summary,
    parse_voice_overrides,
    resolve_voice_plans,
    run_benchmark,
    write_report,
)
from seiyuu.bench.report import collect_environment  # noqa: E402
from seiyuu.bench.runner import DEFAULT_SEED, check_engines, stage_reference  # noqa: E402
from seiyuu.bench.vram import default_probes  # noqa: E402
from seiyuu.gpu import get_gpu_manager  # noqa: E402
from seiyuu.settings import get_settings  # noqa: E402


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--engines", nargs="+", default=list(LOCAL_ENGINES), choices=list(LOCAL_ENGINES),
        help="engines to benchmark, in order (default: all local engines)",
    )  # fmt: skip
    p.add_argument(
        "--voice", action="append", default=[], metavar="ENGINE=VOICE_ID",
        help="voice per engine (repeatable), e.g. --voice kokoro=bf_emma",
    )  # fmt: skip
    p.add_argument(
        "--reference", type=Path, default=None,
        help="reference wav for cloning engines without a --voice (used ephemerally; needs "
        "--consent)",
    )  # fmt: skip
    p.add_argument(
        "--use-test-reference", action="store_true",
        help="fall back to settings.indextts2_test_reference for cloning engines with no voice "
        "(needs --consent)",
    )  # fmt: skip
    p.add_argument(
        "--consent", action="store_true",
        help="attest you have the speaker's consent to clone the --reference / test clip; "
        "applies to this run only and is never persisted",
    )  # fmt: skip
    p.add_argument("--book", default=None, help="take segments from this ingested book (id/prefix)")
    p.add_argument("--chapter", type=int, default=1, help="1-based chapter for --book")
    p.add_argument("--limit", type=int, default=10, help="max blocks taken from --book")
    p.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="seed for preset/--reference voices (library voices use their pinned seed)",
    )  # fmt: skip
    p.add_argument("--warmup", type=int, default=1, help="untimed warm-up synths per engine")
    p.add_argument("--no-determinism", action="store_true", help="skip the same-seed-twice check")
    p.add_argument(
        "--validate", action="store_true",
        help="whisper-validate every timed segment on CPU (slow; after timing)",
    )  # fmt: skip
    p.add_argument("--no-audio", action="store_true", help="don't save the segment wavs")
    p.add_argument("--gpu-index", type=int, default=0, help="nvidia-smi GPU index to sample")
    p.add_argument("--smi-interval", type=float, default=0.2, help="nvidia-smi sample period (s)")
    p.add_argument(
        "--out", type=Path, default=None,
        help="output dir (default: <output_dir>/benchmarks/<timestamp>)",
    )  # fmt: skip
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = get_settings()
    out_dir: Path = args.out or (
        cfg.output_dir / "benchmarks" / datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    from seiyuu.voices import VoiceLibrary

    try:
        engines = check_engines(args.engines)
        overrides = parse_voice_overrides(args.voice)
        if args.warmup < 0:
            raise BenchError(f"--warmup must be >= 0, got {args.warmup}")
        if (args.reference or args.use_test_reference) and not args.consent:
            flag = "--reference" if args.reference else "--use-test-reference"
            raise BenchError(
                f"{flag} clones a voice from an ad-hoc clip; cloning requires --consent (attest "
                "you have the speaker's consent; it applies to this run only and is not saved)"
            )
        normalizer = None
        if args.book:
            from seiyuu.normalize import normalize_text, profile_for
            from seiyuu.normalize.lexicon import load_compiled_lexicon
            from seiyuu.repository import RepositoryError, resolve_book_id

            try:
                book_id = resolve_book_id(args.book, books_dir=cfg.books_dir)
            except RepositoryError as exc:
                raise BenchInputError(str(exc)) from exc
            book_dir = cfg.books_dir / book_id
            segments = book_segments(book_dir, args.chapter, limit=args.limit)
            lexicon = load_compiled_lexicon(book_dir)  # same pronunciations a render would use

            def normalizer(text: str, engine_id: str) -> str:
                return normalize_text(text, profile=profile_for(engine_id), lexicon=lexicon)

            source = f"book:{book_id}:ch{args.chapter}"
        else:
            segments, source = bundled_segments(), "bundled"
    except (BenchError, BenchInputError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="seiyuu-bench-") as scratch:
        try:
            ref_dir = (
                stage_reference(args.reference, Path(scratch) / "voices")
                if args.reference
                else None
            )
        except BenchError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        library = VoiceLibrary(cfg.voices_dir)
        plans = resolve_voice_plans(
            engines, library=library, overrides=overrides, reference_voices_dir=ref_dir,
            kokoro_default=cfg.kokoro_default_voice, seed=args.seed,
            reference_clip=str(args.reference.resolve()) if args.reference else None,
        )  # fmt: skip
        # Last resort for cloning engines with no library clone and no --reference: the
        # configured test clip (settings.indextts2_test_reference), only when asked for
        # (--use-test-reference, with --consent checked above), never silently.
        fallback = cfg.indextts2_test_reference
        fallback_ok = bool(fallback) and Path(fallback).is_file()
        unvoiced = [p for p in plans if p.skip_reason and p.voice_id == "-"]
        if unvoiced and ref_dir is None and args.use_test_reference:
            if not fallback_ok:
                print(
                    f"error: --use-test-reference: no test clip at {fallback} (set "
                    "INDEXTTS2_TEST_REFERENCE)",
                    file=sys.stderr,
                )
                return 2
            print(
                f"no cloned voice for {', '.join(p.engine_id for p in unvoiced)}; using the "
                f"test clip {fallback} (ad-hoc; consent attested for this run only)"
            )
            ref_dir = stage_reference(Path(fallback), Path(scratch) / "voices")
            plans = resolve_voice_plans(
                engines, library=library, overrides=overrides, reference_voices_dir=ref_dir,
                kokoro_default=cfg.kokoro_default_voice, seed=args.seed,
                reference_clip=str(Path(fallback).resolve()), reference_source="test-reference",
            )  # fmt: skip
        elif unvoiced and ref_dir is None and fallback_ok:
            for p in unvoiced:  # point at the opt-in rather than silently cloning the clip
                p.skip_reason = (
                    f"{p.skip_reason}; or --use-test-reference --consent to use {fallback}"
                )
        validator = None
        if args.validate:
            from seiyuu.validate import Validator

            validator = Validator(
                model_size=cfg.validation_model_size,
                device="cpu",  # never contend with the benchmarked engine
                compute_type=cfg.validation_compute_type,
                min_ratio=cfg.validation_min_ratio,
            )

        print(f"segments: {len(segments)} ({source}); engines: {', '.join(engines)}")
        print(f"output:   {out_dir}")
        print("first runs may download multi-GB weights — a slow cold load there is a download")
        gpu = get_gpu_manager()
        try:
            # never raises for GPU contention / unload failure / Ctrl+C: it returns the partial
            # report with `aborted` set, so what was measured is always written below
            report = run_benchmark(
                plans, segments, gpu=gpu, segments_source=source,
                probe_factory=lambda e: default_probes(
                    e, gpu_index=args.gpu_index, smi_interval_s=args.smi_interval
                ),
                **({"normalizer": normalizer} if normalizer else {}),
                warmup=args.warmup, determinism=not args.no_determinism, validator=validator,
                audio_dir=None if args.no_audio else out_dir / "audio", progress=print,
            )  # fmt: skip
        finally:
            try:
                gpu.free_all()  # the one release of the card claimed for the whole run
            except Exception as exc:  # recorded per engine already; don't lose the report
                print(f"warning: final GPU release failed: {type(exc).__name__}: {exc}")

    report.environment = collect_environment(args.gpu_index)
    json_path, md_path = write_report(report, out_dir)
    print()
    print(console_summary(report))
    print(f"\nwrote {json_path}\n      {md_path}")
    return exit_code(report)


def exit_code(report: BenchReport) -> int:
    if report.aborted is AbortKind.INTERRUPTED:
        return 130
    if any(r.status is EngineStatus.FAILED for r in report.results):
        return 1
    if report.aborted is AbortKind.GPU_BUSY:
        return 2
    if not any(r.status is EngineStatus.OK for r in report.results):
        print("nothing was benchmarked: every engine was skipped (see the reasons above)")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
