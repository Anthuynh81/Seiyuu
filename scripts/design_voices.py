"""Design voices from written descriptions with Qwen3-TTS VoiceDesign, then stage them for cloning.

Input is a JSON cast (a list, or {"voices": [...]}) of
    {"voice_id": "...", "description": "gruff old janitor, slow and tired", "sample_text": "...",
     "name": "optional display name"}
For each voice the VoiceDesign checkpoint speaks the sample text --takes times (different seeds)
into <out>/takes/<voice_id>/take<k>.wav. Every take is scored with Seiyuu's whisper check against
the sample text and for pace (words per minute; 140-210 preferred), and the best is picked:
    passes the speech check > pace in range > match score > pace nearest 175 wpm.
Scores are kept in <out>/takes/<voice_id>/scores.json and summarized in <out>/design.md.

The pick is staged as a CLONED voice in <out>/voices/<voice_id>/: reference.wav (the take) and
meta.json (engine qwen3tts, source "qwen_design", consent attesting a synthetic voice with no real
person behind it). A pick that passed the speech check also gets reference.txt (its exact
transcript), bound in the meta so Qwen clones in ICL mode; a flagged pick is staged without one
(x-vector mode). Nothing is added to your voice library: an --out whose voices/ overlaps it is
refused, and so is overwriting a voice this script didn't stage.

To prefer another take, re-run with --pick VOICE=K: existing takes are reused (no GPU), and the
voice is restaged with fresh consent bound to the new clip. (Copying a take over reference.wav by
hand would break the consent hash.) A take is only reused while the cast entry's description and
sample text are the ones it was designed from; edit either and that voice's takes are designed
again. --force regenerates the takes regardless.

--share NEW=SRC stages SRC's chosen clip under NEW as well (one character under two voice ids,
e.g. attribution split the protagonist in two): NEW is not designed, gets SRC's seed, and its
consent is bound to the same clip. A share is remembered in design.json and restaged on every
run, so re-picking SRC carries NEW along; --unshare NEW drops it (NEW is then designed, if it is
in the cast).

GPU job through Seiyuu's GPU manager (one model at a time; refused while another Seiyuu process
holds the card). Ollama must not be holding a model. Whisper runs on the CPU.

Usage:
    uv run python scripts/design_voices.py cast.json --out output/ab/<book>/ch034/designed
    uv run python scripts/design_voices.py cast.json --out ... --pick narrator=2 --pick staff_auto=3
    uv run python scripts/design_voices.py cast.json --out ... --share new_employee=kim_soleum
"""

import argparse
import getpass
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from seiyuu.services.voice_design import (  # noqa: E402 - after the sys.path insert
    CONSENT_TEXT,
    SOURCE,
    WPM_RANGE,
    TakeScore,
    pick_best,
    score_take,
    take_seed,
)


@dataclass(frozen=True)
class DesignSpec:
    voice_id: str
    description: str
    sample_text: str
    name: str


def _say(msg: str) -> None:
    print(msg, flush=True)


def load_specs(path: Path) -> list[DesignSpec]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["voices"] if isinstance(data, dict) else data
    specs = []
    for row in rows:
        vid = row["voice_id"]
        description = row.get("description") or row.get("instruct") or ""
        sample = row.get("sample_text") or row.get("text") or ""
        if not description.strip() or not sample.strip():
            raise ValueError(f"{vid}: needs a description and a sample_text")
        specs.append(DesignSpec(vid, description.strip(), sample.strip(), row.get("name") or vid))
    if len({s.voice_id for s in specs}) != len(specs):
        raise ValueError("duplicate voice_id in the cast file")
    return specs


def parse_picks(values: list[str]) -> dict[str, int]:
    picks = {}
    for value in values:
        vid, sep, k = value.partition("=")
        if not sep or not k.isdigit() or int(k) < 1:
            raise ValueError(f"--pick wants VOICE=K with K >= 1, got {value!r}")
        picks[vid] = int(k)
    return picks


def parse_shares(values: list[str]) -> dict[str, str]:
    shares = {}
    for value in values:
        new, sep, src = value.partition("=")
        if not sep or not new or not src or new == src:
            raise ValueError(f"--share wants NEW=SRC with two different voice ids, got {value!r}")
        shares[new] = src
    if set(shares) & set(shares.values()):
        raise ValueError("--share: a shared voice can't itself be a source")
    return shares


def check_stageable(voices_dir: Path, voice_ids: list[str]) -> None:
    """Refuse to touch a voice this script didn't stage (a real library voice, say)."""
    for vid in voice_ids:
        meta_path = voices_dir / vid / "meta.json"
        if not meta_path.is_file():
            continue
        source = json.loads(meta_path.read_text("utf-8")).get("source")
        if source != SOURCE:
            raise ValueError(
                f"{meta_path} is not a designed voice (source {source!r}); refusing to "
                "overwrite it — point --out at a throwaway dir"
            )


def stage(spec: DesignSpec, chosen: TakeScore, voices_dir: Path) -> Path:
    """Copy the chosen take in as reference.wav, then a consent-bound meta (the hash is taken
    from the staged COPY, like `seiyuu voices clone`).

    A take that PASSED the speech check also gets its transcript (reference.txt) bound in the
    meta (settings.qwen3tts.ref_text_digest), so Qwen clones it in ICL mode and the transcript
    is part of every segment's cache key. A flagged take (dropped or extra words) is staged
    without one: ICL with a transcript that doesn't match the clip is worse than x-vector.
    Refuses to overwrite a voice this script didn't stage (e.g. --out pointed at a library)."""
    from seiyuu.engines.qwen3tts_engine import REF_TEXT_DIGEST_KEY, reference_text_digest
    from seiyuu.voices import VoiceLibrary
    from seiyuu.voices.library import sha256_file
    from seiyuu.voices.models import ConsentAttestation, VoiceKind, VoiceMeta

    lib = VoiceLibrary(voices_dir)
    check_stageable(voices_dir, [spec.voice_id])
    ref = lib.reference_path(spec.voice_id)
    ref.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(chosen.path, ref)
    txt = ref.with_name("reference.txt")
    settings: dict[str, dict[str, float]] = {}
    if chosen.ok:
        txt.write_text(chosen.text + "\n", encoding="utf-8")
        digest = reference_text_digest(sha256_file(ref), chosen.text)
        settings = {"qwen3tts": {REF_TEXT_DIGEST_KEY: digest}}
    else:
        txt.unlink(missing_ok=True)
    lib.save(
        VoiceMeta(
            voice_id=spec.voice_id,
            name=spec.name,
            kind=VoiceKind.CLONED,
            engine="qwen3tts",
            reference_audio="reference.wav",
            seed=chosen.seed,
            consent_attested=True,
            consent=ConsentAttestation(
                attested_by=f"{getpass.getuser()} (design_voices, Qwen3-TTS VoiceDesign)",
                statement=CONSENT_TEXT,
                reference_sha256=sha256_file(ref),
            ),
            source=SOURCE,
            tags=["ab", "designed"],
            settings=settings,
        )
    )
    return ref


def run(
    specs: list[DesignSpec],
    *,
    out: Path,
    takes: int,
    engine: Any,
    validator: Any,
    gpu: Any,
    normalize: Any,
    picks: dict[str, int] | None = None,
    force: bool = False,
    shares: dict[str, str] | None = None,
    unshare: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Design, score, pick and stage every voice. ``engine`` needs ``design_voice``; ``gpu`` is a
    GpuResourceManager (``acquire``/``free_all``); ``normalize(text) -> str``. ``shares`` maps
    NEW -> SRC: NEW is staged with SRC's chosen clip instead of being designed. Shares persist in
    design.json and are restaged on every run (so re-picking SRC carries NEW along) until NEW is
    named in ``unshare``. Returns the summary written to design.json."""
    picks = picks or {}
    summary_path = out / "design.json"
    merged = json.loads(summary_path.read_text("utf-8")) if summary_path.is_file() else {}
    persisted = {vid: e["shared_from"] for vid, e in merged.items() if "shared_from" in e}
    shares = {**persisted, **(shares or {})}
    for vid in unshare or ():
        shares.pop(vid, None)
        if "shared_from" in merged.get(vid, {}):
            del merged[vid]
            # its staged copy of SRC's clip goes too; designing it (if in specs) restages it
            shutil.rmtree(out / "voices" / vid, ignore_errors=True)
    if set(shares) & set(shares.values()):
        raise ValueError(f"--share: a shared voice can't itself be a source: {shares}")
    unknown = sorted(set(picks) - {s.voice_id for s in specs if s.voice_id not in shares})
    if unknown:
        raise ValueError(f"--pick names voices not in the cast file: {unknown}")
    # before any GPU work, so a wrong --out costs nothing
    check_stageable(out / "voices", [s.voice_id for s in specs] + list(shares))
    summary: dict[str, dict[str, Any]] = {}
    acquired = False
    try:
        for spec in specs:
            if spec.voice_id in shares:
                continue
            text = normalize(spec.sample_text)
            take_dir = out / "takes" / spec.voice_id
            take_dir.mkdir(parents=True, exist_ok=True)
            scores_path = take_dir / "scores.json"
            old: dict[int, dict] = {}
            if scores_path.is_file() and not force:
                saved = json.loads(scores_path.read_text("utf-8"))
                # A take is only as good as the description it was designed from: a changed
                # description makes every take stale (its provenance would be wrong).
                if saved.get("description") == spec.description:
                    old = {s["take"]: s for s in saved["takes"]}
            results: list[TakeScore] = []
            for k in range(1, takes + 1):
                seed = take_seed(spec.voice_id, k)
                path = take_dir / f"take{k}.wav"
                prev = old.get(k)
                # Reuse a take only when its scores record THIS seed, text and description: a
                # take speaking an older sample text would be staged under a transcript it
                # never said. Anything else (incl. a wav with no record) is designed afresh.
                if (
                    prev is not None
                    and path.is_file()
                    and prev.get("seed") == seed
                    and prev.get("text") == text
                ):
                    results.append(TakeScore(**prev))
                else:
                    _say(f"  {spec.voice_id} take {k}: designing (seed {seed})")
                    with gpu.acquire(engine, "qwen3tts"):
                        acquired = True
                        audio = engine.design_voice(text, spec.description, seed=seed)
                    audio.save(path)
                    results.append(score_take(validator, path, text, k, seed))
                # Saved after every take, so an interrupted run keeps what it designed.
                scores_path.write_text(
                    json.dumps(
                        {"description": spec.description, "takes": [asdict(r) for r in results]},
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            if spec.voice_id in picks:
                k = picks[spec.voice_id]
                if not 1 <= k <= len(results):
                    raise ValueError(f"{spec.voice_id}: --pick {k} but only {len(results)} takes")
                chosen, how = results[k - 1], "picked"
            else:
                chosen, how = pick_best(results), "auto"
            stage(spec, chosen, out / "voices")
            _say(
                f"  {spec.voice_id}: take {chosen.take} ({how}) — {chosen.seconds:.1f} s, "
                f"{chosen.wpm:.0f} wpm, match {chosen.score:.2f}{'' if chosen.ok else ' FLAGGED'}"
            )
            summary[spec.voice_id] = {
                "description": spec.description,
                "chosen": chosen.take,
                "how": how,
                "takes": [asdict(r) for r in results],
            }
    finally:
        if acquired:
            gpu.free_all()
    # Merge into an earlier run's summary, so designing a subset (--only) keeps the other voices.
    merged.update(summary)
    names = {s.voice_id: s.name for s in specs}
    for new, src in shares.items():
        entry = merged.get(src)
        if entry is None or "shared_from" in entry:
            raise ValueError(f"--share {new}={src}: {src} has not been designed in {out}")
        chosen = TakeScore(**entry["takes"][entry["chosen"] - 1])
        name = names.get(new) or merged.get(new, {}).get("name") or new
        spec = DesignSpec(new, entry["description"], chosen.text, name)
        stage(spec, chosen, out / "voices")
        _say(f"  {new}: same clip as {src} (take {chosen.take})")
        summary[new] = merged[new] = {
            "description": entry["description"],
            "chosen": chosen.take,
            "how": f"same as {src}",
            "shared_from": src,
            "name": name,
            "takes": [asdict(chosen)],
        }
    summary_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    (out / "design.md").write_text(report(merged), encoding="utf-8")
    return summary


def report(summary: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# Designed voices (Qwen3-TTS VoiceDesign)",
        "",
        "| Voice | Take | Seconds | WPM | Match | Speech check | Chosen |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for vid, entry in summary.items():
        if "shared_from" in entry:
            lines.append(f"| {vid} | — | | | | | {entry['how']}, take {entry['chosen']} |")
            continue
        for t in entry["takes"]:
            mark = f"yes ({entry['how']})" if t["take"] == entry["chosen"] else ""
            check = "ok" if t["ok"] else f"flagged: {t['reason'] or 'mismatch'}"
            lines.append(
                f"| {vid} | {t['take']} | {t['seconds']:.1f} | {t['wpm']:.0f} | "
                f"{t['score']:.2f} | {check} | {mark} |"
            )
    lines += [
        "",
        f"Preferred pace {WPM_RANGE[0]:.0f}-{WPM_RANGE[1]:.0f} wpm. To use another take, re-run "
        "with --pick VOICE=K (reuses the takes; restages with fresh consent).",
        "",
        "Descriptions:",
        "",
        *(
            f"- **{vid}**: {entry['description']}"
            for vid, entry in summary.items()
            if "shared_from" not in entry
        ),
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("cast", type=Path, help="JSON: voice_id, description, sample_text (, name)")
    ap.add_argument("--out", type=Path, required=True, help="takes/, voices/ and design.md go here")
    ap.add_argument("--takes", type=int, default=3, help="takes per voice (different seeds)")
    ap.add_argument("--pick", action="append", default=[], metavar="VOICE=K")
    ap.add_argument("--force", action="store_true", help="regenerate takes that already exist")
    ap.add_argument("--only", nargs="+", default=None, help="design just these voice ids")
    ap.add_argument(
        "--share", action="append", default=[], metavar="NEW=SRC", help="stage SRC's clip as NEW"
    )
    ap.add_argument(
        "--unshare", action="append", default=[], metavar="NEW", help="drop an earlier --share"
    )
    args = ap.parse_args(argv)
    if args.takes < 1:
        ap.error("--takes must be at least 1")

    from seiyuu.settings import get_settings

    cfg = get_settings()
    staged = (args.out / "voices").resolve()
    library = Path(cfg.voices_dir).resolve()
    if staged == library or library in staged.parents or staged in library.parents:
        ap.error(f"--out stages voices into {staged}, which overlaps your voice library {library}")
    specs = load_specs(args.cast)
    shares = parse_shares(args.share)
    picks = parse_picks(args.pick)
    cast = {s.voice_id for s in specs}
    if args.only:
        only = set(args.only)
        if only - cast:
            ap.error(f"--only names voices not in the cast file: {sorted(only - cast)}")
        if set(picks) & cast - only:
            left_out = sorted(set(picks) & cast - only)
            ap.error(f"--pick names voices that --only leaves out: {left_out}")
        specs = [s for s in specs if s.voice_id in only or s.voice_id in shares]

    from seiyuu.engines.qwen3tts_engine import Qwen3TTSEngine
    from seiyuu.gpu import get_gpu_manager
    from seiyuu.normalize import normalize_text
    from seiyuu.validate import Validator

    args.out.mkdir(parents=True, exist_ok=True)
    validator = Validator(
        model_size=cfg.validation_model_size,
        device=cfg.whisper_device,
        compute_type=cfg.validation_compute_type,
        min_ratio=cfg.validation_min_ratio,
    )
    engine = Qwen3TTSEngine(voices_dir=args.out / "voices")
    _say(f"designing {len(specs)} voice(s), {args.takes} take(s) each -> {args.out}")
    run(
        specs,
        out=args.out,
        takes=args.takes,
        engine=engine,
        validator=validator,
        gpu=get_gpu_manager(),
        normalize=lambda t: normalize_text(t, profile="qwen3tts"),
        picks=picks,
        force=args.force,
        shares=shares,
        unshare=set(args.unshare),
    )
    _say(f"staged voices: {args.out / 'voices'}; report: {args.out / 'design.md'}")
    _say(f"worker restarts: {engine.worker_restarts}, capped takes: {engine.capped_lines}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
