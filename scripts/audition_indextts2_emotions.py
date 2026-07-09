"""Render an IndexTTS-2 emotion-audition matrix for tuning voices/emotion.py by ear (M7 polish).

Boots the real subprocess worker ONCE (model load is the expensive part), renders one clip per
requested (emotion, intensity) cell — each through the SAME ``map_emotion`` mapping the render
pipeline uses — plus a neutral baseline, and writes labeled WAVs to ``output/emotion_audition/``.
Listen, then tune ``_INDEXTTS2_WEIGHT`` / ``_INDEXTTS2_ALPHA`` / the per-label dims in
``voices/emotion.py`` (changing them re-renders emotive segments only; NEUTRAL stays byte-stable).

The reference voice comes from --reference, or settings.indextts2_test_reference (.env).

Usage (GPU job — each clip is slow; see the printed estimate):
    uv run python scripts/audition_indextts2_emotions.py                # 6 labels @ i2 + neutral
    uv run python scripts/audition_indextts2_emotions.py --intensities 1 2 3    # full matrix
    uv run python scripts/audition_indextts2_emotions.py --labels angry happy   # subset
    uv run python scripts/audition_indextts2_emotions.py --sweep angry      # alpha x weight sweep
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from seiyuu.attribute.models import EmotionLabel, EmotionVerdict  # noqa: E402
from seiyuu.engines import get_engine  # noqa: E402
from seiyuu.settings import get_settings  # noqa: E402
from seiyuu.voices.emotion import map_emotion  # noqa: E402

# One emotion-appropriate line per label, so the delivery can be judged naturally. Same length
# ballpark (~4 s) keeps per-clip render time comparable.
_LINES: dict[EmotionLabel, str] = {
    EmotionLabel.HAPPY: "We actually did it! I can't stop smiling; this is the best news all year.",
    EmotionLabel.SAD: "She kept his letters in a box she could no longer bring herself to open.",
    EmotionLabel.ANGRY: "You knew the bridge was failing, and you sent them across it anyway!",
    EmotionLabel.FEARFUL: "Something just moved in the dark hallway. Tell me you heard it too.",
    EmotionLabel.TENDER: "It's all right, little one. I'm here now, and I'm not going anywhere.",
    EmotionLabel.TENSE: "Don't turn around. Keep walking, keep your voice level, and stay with me.",
}
_NEUTRAL_LINE = "The library closed at nine, and the last train left the station a half hour later."


def _resolve_reference() -> Path:
    cfg = get_settings()
    ref = cfg.indextts2_test_reference
    if not ref or not Path(ref).is_file():
        raise SystemExit(
            "no reference clip: set indextts2_test_reference in .env (or pass --reference)"
        )
    return Path(ref)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--labels", nargs="*", default=[label.value for label in _LINES],
        choices=[label.value for label in _LINES], help="emotion labels to render",
    )  # fmt: skip
    parser.add_argument(
        "--intensities", nargs="*", type=int, default=[2], choices=[1, 2, 3],
        help="intensity levels per label (default: 2 = medium only)",
    )  # fmt: skip
    parser.add_argument(
        "--sweep", default=None, choices=[label.value for label in _LINES],
        help="instead of the matrix: sweep emo_alpha x dominant weight for ONE label",
    )  # fmt: skip
    parser.add_argument("--reference", type=Path, default=None, help="reference clip override")
    parser.add_argument("--seed", type=int, default=41172)
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "output" / "emotion_audition",
        help="output directory",
    )  # fmt: skip
    args = parser.parse_args()

    reference = args.reference or _resolve_reference()
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # A throwaway voice dir so the engine resolves the reference the normal way.
    voices_dir = out_dir / "_voices"
    (voices_dir / "audition").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(reference, voices_dir / "audition" / "reference.wav")

    # Build the work list: (clip name, text, settings-overrides).
    jobs: list[tuple[str, str, dict]] = [("neutral", _NEUTRAL_LINE, {})]
    if args.sweep:
        label = EmotionLabel(args.sweep)
        text = _LINES[label]
        base = map_emotion("indextts2", EmotionVerdict(label=label, intensity=2))
        dim = next(i for i, v in enumerate(base["emo_vector"]) if v > 0)
        for weight in (0.6, 0.8, 1.0):
            for alpha in (0.6, 0.8, 1.0):
                vector = [0.0] * 8
                vector[dim] = weight
                jobs.append(
                    (
                        f"{label.value}_w{weight:.1f}_a{alpha:.1f}",
                        text,
                        {"emo_vector": vector, "emo_alpha": alpha},
                    )
                )
    else:
        for label_name in args.labels:
            label = EmotionLabel(label_name)
            for intensity in args.intensities:
                override = map_emotion(
                    "indextts2", EmotionVerdict(label=label, intensity=intensity)
                )
                jobs.append((f"{label.value}_i{intensity}", _LINES[label], override))

    print(f"reference: {reference}")
    print(f"output:    {out_dir}")
    print(f"clips:     {len(jobs)} (model loads once; on the sysmem-fallback path expect")
    print("           roughly 1.5-2.5 min per clip; ~10x faster with fallback OFF)")

    engine = get_engine("indextts2", voices_dir=voices_dir)
    try:
        for i, (name, text, override) in enumerate(jobs, start=1):
            start = time.monotonic()
            audio = engine.synthesize(text, "audition", {"seed": args.seed, **override})
            path = audio.save(out_dir / f"{name}.wav")
            took = time.monotonic() - start
            rtf = took / max(audio.duration_seconds, 1e-9)
            print(
                f"[{i}/{len(jobs)}] {name}: {audio.duration_seconds:.1f}s audio "
                f"in {took:.0f}s (RTF {rtf:.1f}x) -> {path.name}"
            )
    finally:
        engine.unload()  # terminate the worker -> reclaim VRAM
        shutil.rmtree(voices_dir, ignore_errors=True)

    print("\nListen in order (neutral first as the baseline). For each emotive clip ask:")
    print("  1. is the emotion recognizable?  2. too weak / about right / overacted?")
    print("  3. any artifacts (rushing, slurring, pitch weirdness) vs neutral?")
    print("Then tune _INDEXTTS2_WEIGHT / _INDEXTTS2_ALPHA / dims in src/seiyuu/voices/emotion.py.")


if __name__ == "__main__":
    main()
