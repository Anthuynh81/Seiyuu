"""IndexTTS-2 subprocess worker — runs in the SEPARATE cu128 uv env, NOT this venv.

IndexTTS-2 hard-pins torch 2.8/cu128 + transformers 4.52, which conflict irreconcilably with
this project's torch 2.6 + transformers 5 (chatterbox's pins). A single interpreter can hold
only one of each, so IndexTTS-2 runs out-of-process here and ``indextts2_engine.py`` (in the
main venv) drives it over newline-delimited JSON on stdin/stdout.

HARD RULE: this file is executed as a STANDALONE SCRIPT (``python indextts2_worker.py``) by an
interpreter that has the IndexTTS-2 SDK installed and does NOT have seiyuu (or its torch 2.6).
So it must NEVER import seiyuu, and it keeps module-top imports to the stdlib only — torch and
the IndexTTS-2 SDK are imported lazily, the first time a model actually loads.

Protocol (one JSON object per line, request -> reply):
- boot: the worker emits ``{"ok": true, "event": "ready", "protocol": N}`` once, unprompted,
  after its own process starts (before any weights load) so the adapter can confirm it booted.
- ``{"cmd": "ping"}``       -> ``{"ok": true, "protocol": N}``
- ``{"cmd": "load"}``       -> ``{"ok": true}`` (loads/keeps the model resident)
- ``{"cmd": "synthesize", "text", "reference_wav", "out_path", "seed"?, "emo_vector"?,
     "emo_alpha"?, "gen"?}`` -> ``{"ok": true, "sample_rate": 22050, "path": out_path}``
  (``gen`` = adapter-whitelisted infer() tunables, e.g. temperature/top_p; identity/output
  args are protected and stripped — see ``_PROTECTED_INFER_KWARGS``.)
- any failure -> ``{"ok": false, "error": str, "oom": bool, "traceback": str}``. On a CUDA OOM
  the worker frees what it can and flags ``oom`` so the adapter kills+restarts it (process
  death is the only reliable VRAM reclaim at the 8GB ceiling).
- EOF on stdin ends the loop (the adapter closes stdin to shut the worker down).

stdout MUST stay pure JSON: the IndexTTS-2 SDK prints progress, so every model call is wrapped
in ``redirect_stdout(sys.stderr)`` — library chatter goes to stderr, the JSON channel stays clean.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from typing import Any, TextIO

# Bumped when the request/reply shape changes; the adapter checks it at boot.
PROTOCOL_VERSION = 1

# IndexTTS-2 is hardcoded to 22050 Hz (confirmed against infer_v2.py); the adapter's
# to_canonical() resamples to the project's canonical 24 kHz.
SAMPLE_RATE = 22050

# infer() arguments the adapter's `gen` tunables may never override — the worker owns these
# (identity/emotion/output/determinism). The adapter whitelists on its side; this is the
# defensive second layer so a malformed message can't hijack the reference or seed discipline.
_PROTECTED_INFER_KWARGS = frozenset(
    {
        "spk_audio_prompt", "text", "output_path", "emo_audio_prompt", "emo_alpha", "emo_vector",
        "use_emo_text", "emo_text", "use_random", "verbose", "stream_return",
    }
)  # fmt: skip


def _is_oom(exc: BaseException) -> bool:
    """True for a CUDA out-of-memory error, however torch spells it across versions."""
    return "OutOfMemory" in type(exc).__name__ or "out of memory" in str(exc).lower()


class ModelHandle:
    """Lazily loads IndexTTS-2 and runs inference. The SDK/torch imports live here so the
    worker process can boot (and answer ``ping``) before paying the multi-GB load."""

    def __init__(self, checkpoints_dir: str, use_fp16: bool) -> None:
        self._checkpoints_dir = checkpoints_dir
        self._use_fp16 = use_fp16
        self._tts: Any | None = None

    def load(self) -> None:
        if self._tts is not None:
            return
        import os
        from contextlib import redirect_stdout

        # Keep stdout pure JSON: the SDK banner/progress prints go to stderr.
        with redirect_stdout(sys.stderr):
            from indextts.infer_v2 import IndexTTS2  # SDK import stays inside the worker

            cfg_path = os.path.join(self._checkpoints_dir, "config.yaml")
            # use_cuda_kernel=False: the BigVGAN CUDA kernel build fails on Windows (see scope);
            # use_deepspeed omitted for the same reason (installed WITHOUT --all-extras).
            self._tts = IndexTTS2(
                cfg_path=cfg_path,
                model_dir=self._checkpoints_dir,
                use_fp16=self._use_fp16,
                use_cuda_kernel=False,
            )

    def synthesize(
        self,
        *,
        text: str,
        reference_wav: str,
        out_path: str,
        seed: int | None,
        emo_vector: list[float] | None,
        emo_alpha: float | None,
        gen: dict[str, Any] | None = None,
    ) -> int:
        from contextlib import redirect_stdout

        import torch

        self.load()
        with redirect_stdout(sys.stderr):
            if seed is not None:
                # infer() has no seed arg and samples (do_sample=True); seed right before it or
                # identical SegmentKeys would map to different audio (Chatterbox does the same).
                torch.manual_seed(int(seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(seed))
            kwargs: dict[str, Any] = {}
            if gen:  # adapter-whitelisted tunables; never the worker-owned identity args
                kwargs.update({k: v for k, v in gen.items() if k not in _PROTECTED_INFER_KWARGS})
            if emo_vector is not None:
                kwargs["emo_vector"] = [float(x) for x in emo_vector]
            if emo_alpha is not None:
                kwargs["emo_alpha"] = float(emo_alpha)
            # use_random=False: no extra sampling entropy on top of the seed (reproducibility).
            self._tts.infer(  # type: ignore[union-attr]
                spk_audio_prompt=reference_wav,
                text=text,
                output_path=out_path,
                use_random=False,
                verbose=False,
                **kwargs,
            )
        return SAMPLE_RATE

    def empty_cache(self) -> None:
        """Best-effort VRAM reclaim after an OOM (the adapter still kills+restarts us)."""
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 — cleanup must never mask the original error
            pass


def handle(msg: dict[str, Any], model: ModelHandle) -> dict[str, Any]:
    """Dispatch one decoded request to a reply dict. Pure of I/O so it is unit-testable
    with a fake model; ``serve`` owns the stdin/stdout framing and error wrapping."""
    cmd = msg.get("cmd")
    if cmd == "ping":
        return {"ok": True, "protocol": PROTOCOL_VERSION}
    if cmd == "load":
        model.load()
        return {"ok": True}
    if cmd == "synthesize":
        sample_rate = model.synthesize(
            text=msg["text"],
            reference_wav=msg["reference_wav"],
            out_path=msg["out_path"],
            seed=msg.get("seed"),
            emo_vector=msg.get("emo_vector"),
            emo_alpha=msg.get("emo_alpha"),
            gen=msg.get("gen"),
        )
        return {"ok": True, "sample_rate": sample_rate, "path": msg["out_path"]}
    return {"ok": False, "error": f"unknown cmd {cmd!r}", "oom": False}


def _write(stream: TextIO, obj: dict[str, Any]) -> None:
    stream.write(json.dumps(obj) + "\n")
    stream.flush()


def serve(stdin: TextIO, stdout: TextIO, model: ModelHandle) -> None:
    """Read newline-JSON requests until EOF, writing one reply per request. Every exception
    becomes a structured ``{"ok": false, ...}`` reply (with an ``oom`` flag) so a single bad
    segment never takes the worker down silently."""
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError as exc:
            _write(stdout, {"ok": False, "error": f"bad json: {exc}", "oom": False})
            continue
        try:
            reply = handle(msg, model)
        except Exception as exc:  # noqa: BLE001 — every failure is reported, never crashes serve
            oom = _is_oom(exc)
            if oom:
                model.empty_cache()
            reply = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "oom": oom,
                "traceback": traceback.format_exc(),
            }
        _write(stdout, reply)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="IndexTTS-2 stdio worker")
    parser.add_argument("--checkpoints", required=True, help="IndexTTS-2 checkpoints dir")
    parser.add_argument("--fp16", dest="fp16", action="store_true", help="load in fp16 (8GB fit)")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.set_defaults(fp16=True)
    args = parser.parse_args(argv)

    model = ModelHandle(args.checkpoints, args.fp16)
    # Announce readiness on the pure-JSON channel (process booted; weights load lazily on `load`).
    _write(sys.stdout, {"ok": True, "event": "ready", "protocol": PROTOCOL_VERSION})
    serve(sys.stdin, sys.stdout, model)


if __name__ == "__main__":
    main()
