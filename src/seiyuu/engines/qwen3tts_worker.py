"""Qwen3-TTS subprocess worker — runs in the SEPARATE Qwen3-TTS uv env, NOT this venv.

Qwen3-TTS (via faster-qwen3-tts on qwen-tts-hf) pins torch 2.8/cu128 + transformers 5.15.1, which
conflict with this project's torch 2.6 (chatterbox's pin). So, like IndexTTS-2, it runs
out-of-process here and ``qwen3tts_engine.py`` (in the main venv) drives it over newline-delimited
JSON on stdin/stdout.

HARD RULE: this file is executed as a STANDALONE SCRIPT (``python qwen3tts_worker.py``) by an
interpreter that has the Qwen3-TTS packages and does NOT have seiyuu. So it must NEVER import
seiyuu, and it keeps module-top imports to the stdlib only — torch, numpy, soundfile and the SDK
are imported lazily, the first time a model actually loads.

Two checkpoints, ONE resident at a time (GPU discipline inside the worker too):
- ``base`` (Qwen3-TTS-12Hz-1.7B-Base) clones a voice from its reference clip — every render.
- ``design`` (Qwen3-TTS-12Hz-1.7B-VoiceDesign) speaks text in a voice described in words — only
  the one-off voice-design step. Asking for the other checkpoint unloads the resident one first.

Protocol (one JSON object per line, request -> reply):
- boot: the worker emits ``{"ok": true, "event": "ready", "protocol": N}`` once, unprompted.
- ``{"cmd": "ping"}``  -> ``{"ok": true, "protocol": N}``
- ``{"cmd": "load", "model"?: "base"|"design"}`` -> ``{"ok": true, "model": ...}``
- ``{"cmd": "synthesize", "text", "reference_wav", "out_path", "reference_sha256"?, "ref_text"?,
     "seed"?, "gen"?}`` -> ``{"ok": true, "sample_rate": 24000, "path", "seconds", "hit_cap"}``
  Clone with the Base checkpoint. ``ref_text`` (the reference clip's exact transcript) selects
  ICL mode; without it the speaker embedding alone is used (x-vector mode, weaker likeness).
- ``{"cmd": "design", "text", "instruct", "out_path", "seed"?, "gen"?}`` -> same reply shape.
- any failure -> ``{"ok": false, "error": str, "oom": bool, "traceback": str}``.
- EOF on stdin ends the loop (the adapter closes stdin to shut the worker down).

stdout MUST stay pure JSON. Model calls run under ``redirect_stdout(sys.stderr)``, and when run as
a script the worker also moves file descriptor 1 onto stderr (keeping a private dup for JSON), so
even C-level or subprocess prints from the SDK can't corrupt the channel.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from typing import Any, TextIO

# Bumped when the request/reply shape changes; the adapter checks it at boot.
PROTOCOL_VERSION = 1

# The 12 Hz codec decodes to 24 kHz (1920 samples per frame -> 12.5 frames per second). That is
# already Seiyuu's canonical rate, so the adapter's to_canonical() never resamples.
SAMPLE_RATE = 24000
FRAMES_PER_SECOND = 12.5

CHECKPOINTS = {
    "base": "Qwen3-TTS-12Hz-1.7B-Base",
    "design": "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}

# Sampling tunables the adapter may forward (talker only; the 15-codebook predictor samples with
# values fixed when the model is built). Anything else in `gen` is ignored, so a malformed message
# can't override the text, reference, output or determinism args.
_GEN_KEYS = frozenset({"temperature", "top_p", "top_k", "repetition_penalty"})
_INT_GEN_KEYS = frozenset({"top_k"})  # VoiceMeta.settings stores floats; the sampler needs int

# Quote glyphs a dialogue line may open/close with. Upstream issue #80: lines that START with a
# quote mark are the ones most often spoken in the wrong voice. The glyphs are never spoken, so
# stripping the wrapping pair costs nothing (validation still compares against the full normalized
# text; the scorer ignores punctuation). Single quotes are left alone: a leading ' is often an
# elision ('Cause, 'em) rather than a quote.
_OPEN_QUOTES = '"“”„«»「『'
_CLOSE_QUOTES = '"“”«»」』'

# Seconds of silence appended to the reference before encoding it (ICL mode). The prompt ends on
# the reference's last codec frames, so padding keeps its final phoneme from bleeding into the
# first generated word — what faster-qwen3-tts's own reference path does (append_silence=True).
_REFERENCE_PAD_SECONDS = 0.5


def model_text(text: str) -> str:
    """The text actually sent to the model: a line wrapped in double quotes loses the quotes."""
    stripped = text.strip()
    if not stripped or stripped[0] not in _OPEN_QUOTES:
        return text
    inner = stripped.lstrip(_OPEN_QUOTES).rstrip().rstrip(_CLOSE_QUOTES).strip()
    return inner or text


# Generation cap calibration (3080 Ti smoke, 2026-11-07): designed and cloned English speech ran
# 17-22 characters/s, and a non-streaming ICL prompt measured 9 + reference frames + ~4.5
# characters per text token (both texts are prefilled). The cap allows 4 s plus one second per 8
# characters (over 2x the measured duration of any line past a few words), and the prompt estimate
# rounds tokens up (3 characters each) so it never undershoots.
_CAP_FLOOR_SECONDS = 4.0
_CAP_CHARS_PER_SECOND = 8.0
_CAP_MIN_FRAMES = 64
_PROMPT_OVERHEAD_TOKENS = 16
_PROMPT_CHARS_PER_TOKEN = 3.0


def max_new_tokens_for(
    text: str, max_seq_len: int, *, prompt_frames: int = 0, prompt_chars: int = 0
) -> int:
    """Generation cap for one line: a runaway (looping, babbling) never runs to the 2048 default.

    ``prompt_frames`` (the ICL reference's codec frames) and ``prompt_chars`` (the reference
    transcript or the design description) size the prompt that shares the static KV cache. The
    fast path stops SILENTLY when prompt + generation reaches ``max_seq_len``; bounding the cap by
    the room left keeps the cap the binding stop, so ``hit_cap`` stays truthful.
    """
    seconds = _CAP_FLOOR_SECONDS + len(text) / _CAP_CHARS_PER_SECOND
    frames = math.ceil(seconds * FRAMES_PER_SECOND)
    prompt = (
        _PROMPT_OVERHEAD_TOKENS
        + prompt_frames
        + math.ceil((prompt_chars + len(text)) / _PROMPT_CHARS_PER_TOKEN)
    )
    return max(_CAP_MIN_FRAMES, min(frames, max_seq_len - 1 - prompt))


def _is_oom(exc: BaseException) -> bool:
    """True for a CUDA out-of-memory error, however torch spells it across versions."""
    return "OutOfMemory" in type(exc).__name__ or "out of memory" in str(exc).lower()


class ModelHandle:
    """Lazily loads ONE Qwen3-TTS checkpoint and runs inference. The SDK/torch imports live here
    so the worker process can boot (and answer ``ping``) before paying the multi-GB load."""

    def __init__(
        self,
        models_dir: str,
        *,
        backend: str = "fast",
        language: str = "English",
        max_seq_len: int = 2048,
    ) -> None:
        self._models_dir = models_dir
        self._backend = backend
        self._language = language
        self._max_seq_len = max_seq_len
        self._tts: Any | None = None
        self.kind: str | None = None  # which checkpoint is resident
        # (reference sha256 or path, ref_text, x-vector?) -> voice-clone prompt items. Keyed by the
        # reference's CONTENT hash, so a re-cloned voice (same path, new audio) is re-encoded.
        self._prompts: dict[tuple[str, str, bool], Any] = {}

    def load(self, kind: str = "base") -> None:
        if kind not in CHECKPOINTS:
            raise ValueError(f"unknown checkpoint {kind!r}; expected one of {sorted(CHECKPOINTS)}")
        if self._tts is not None and self.kind == kind:
            return
        self.unload()  # one checkpoint resident at a time
        from contextlib import redirect_stdout

        with redirect_stdout(sys.stderr):
            import torch

            path = os.path.join(self._models_dir, CHECKPOINTS[kind])
            if self._backend == "stock":
                # The reference package: no CUDA graphs, slower, but the upstream code path.
                from qwen_tts import Qwen3TTSModel

                self._tts = Qwen3TTSModel.from_pretrained(
                    path, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa"
                )
            else:
                # faster-qwen3-tts: the same model with CUDA-graph decoding. Needs a CUDA device.
                # A local dir also makes it load the bundled speech_tokenizer/ (no hub download).
                from faster_qwen3_tts import FasterQwen3TTS

                self._tts = FasterQwen3TTS.from_pretrained(
                    model_name=path,
                    device="cuda",
                    dtype=torch.bfloat16,
                    attn_implementation="sdpa",
                    max_seq_len=self._max_seq_len,
                )
                # Capture the CUDA graphs now: inside the load timeout rather than the first
                # line's, and never between that line's seeding and its sampling.
                self._tts.warmup()
        self.kind = kind

    def unload(self) -> None:
        if self._tts is None:
            return
        self._tts = None
        self.kind = None
        self._prompts.clear()  # the cached prompts hold GPU tensors of the old model
        self.empty_cache()

    @property
    def _qwen(self) -> Any:
        """The underlying ``qwen_tts.Qwen3TTSModel`` (the fast wrapper keeps it at ``.model``)."""
        return self._tts if self._backend == "stock" else self._tts.model  # type: ignore[union-attr]

    def _gen_kwargs(
        self,
        text: str,
        gen: dict[str, Any] | None,
        *,
        prompt_frames: int = 0,
        prompt_chars: int = 0,
    ) -> dict[str, Any]:
        cap = max_new_tokens_for(
            text, self._max_seq_len, prompt_frames=prompt_frames, prompt_chars=prompt_chars
        )
        kwargs: dict[str, Any] = {"max_new_tokens": cap}
        for k, v in (gen or {}).items():
            if k in _GEN_KEYS:
                kwargs[k] = int(v) if k in _INT_GEN_KEYS else float(v)
        return kwargs

    def _clone_prompt(
        self, reference_wav: str, reference_sha256: str | None, ref_text: str
    ) -> tuple[Any, int]:
        """The encoded clone prompt and how many codec frames of reference it puts in the
        prompt (0 in x-vector mode; the padded clip's length in ICL mode)."""
        xvec = not ref_text
        key = (reference_sha256 or reference_wav, ref_text, xvec)
        cached = self._prompts.get(key)
        if cached is not None:
            return cached
        frames = 0
        if xvec:
            # x-vector mode: speaker embedding only, no transcript, no reference codes.
            prompt = self._qwen.create_voice_clone_prompt(
                ref_audio=reference_wav, ref_text=None, x_vector_only_mode=True
            )
        else:
            import numpy as np
            import soundfile as sf

            audio, sr = sf.read(reference_wav, dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            padded = np.concatenate([audio, np.zeros(int(_REFERENCE_PAD_SECONDS * sr), np.float32)])
            prompt = self._qwen.create_voice_clone_prompt(
                ref_audio=(padded, int(sr)),
                ref_text=ref_text,
                x_vector_only_mode=False,
            )
            frames = math.ceil(padded.size / float(sr) * FRAMES_PER_SECOND)
        self._prompts[key] = (prompt, frames)
        return prompt, frames

    @staticmethod
    def _seed(seed: int | None) -> None:
        if seed is None:
            return
        import torch

        # No Qwen3-TTS API takes a seed and every step samples (torch.multinomial on the default
        # generators), so seed right before each call or identical SegmentKeys would map to
        # different audio. manual_seed seeds the CPU and every CUDA device.
        torch.manual_seed(int(seed))

    def synthesize(
        self,
        *,
        text: str,
        reference_wav: str,
        out_path: str,
        reference_sha256: str | None = None,
        ref_text: str | None = None,
        seed: int | None = None,
        gen: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        from contextlib import redirect_stdout

        self.load("base")
        with redirect_stdout(sys.stderr):
            ref_text = (ref_text or "").strip()
            prompt, ref_frames = self._clone_prompt(reference_wav, reference_sha256, ref_text)
            kwargs = self._gen_kwargs(
                text, gen, prompt_frames=ref_frames, prompt_chars=len(ref_text)
            )
            self._seed(seed)
            # non_streaming_mode=True prefills the whole line before decoding. The clone default
            # (False) feeds text step by step, which drifts toward rushing on longer lines.
            wavs, sr = self._tts.generate_voice_clone(  # type: ignore[union-attr]
                text=text,
                language=self._language,
                voice_clone_prompt=prompt,
                non_streaming_mode=True,
                **kwargs,
            )
        return _write_wav(out_path, wavs[0], int(sr), kwargs["max_new_tokens"])

    def design(
        self,
        *,
        text: str,
        instruct: str,
        out_path: str,
        seed: int | None = None,
        gen: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        from contextlib import redirect_stdout

        self.load("design")
        with redirect_stdout(sys.stderr):
            kwargs = self._gen_kwargs(text, gen, prompt_chars=len(instruct))
            self._seed(seed)
            wavs, sr = self._tts.generate_voice_design(  # type: ignore[union-attr]
                text=text,
                instruct=instruct,
                language=self._language,
                non_streaming_mode=True,
                **kwargs,
            )
        return _write_wav(out_path, wavs[0], int(sr), kwargs["max_new_tokens"])

    def empty_cache(self) -> None:
        """Best-effort VRAM reclaim (after an OOM or a checkpoint swap)."""
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 — cleanup must never mask the original error
            pass


def _write_wav(out_path: str, wav: Any, sample_rate: int, max_new_tokens: int) -> dict[str, Any]:
    """Write the model's float waveform as 16-bit mono PCM and describe it for the reply.

    A generation that produced no tokens comes back as a single zero sample; it is written as an
    EMPTY file so the adapter fails it loudly instead of shipping a click as speech."""
    import numpy as np
    import soundfile as sf

    samples = np.asarray(wav, dtype=np.float32).reshape(-1)
    if samples.size <= 1:
        samples = np.zeros(0, dtype=np.float32)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    sf.write(out_path, pcm, sample_rate, subtype="PCM_16")
    seconds = samples.size / float(sample_rate)
    # Within one frame of the cap: the model most likely never emitted its end token.
    hit_cap = seconds * FRAMES_PER_SECOND >= max_new_tokens - 1
    return {"sample_rate": sample_rate, "seconds": round(seconds, 3), "hit_cap": hit_cap}


def handle(msg: dict[str, Any], model: ModelHandle) -> dict[str, Any]:
    """Dispatch one decoded request to a reply dict. Pure of I/O so it is unit-testable
    with a fake model; ``serve`` owns the stdin/stdout framing and error wrapping."""
    cmd = msg.get("cmd")
    if cmd == "ping":
        return {"ok": True, "protocol": PROTOCOL_VERSION}
    if cmd == "load":
        kind = msg.get("model", "base")
        model.load(kind)
        return {"ok": True, "model": kind}
    if cmd == "synthesize":
        info = model.synthesize(
            text=model_text(msg["text"]),
            reference_wav=msg["reference_wav"],
            out_path=msg["out_path"],
            reference_sha256=msg.get("reference_sha256"),
            ref_text=msg.get("ref_text"),
            seed=msg.get("seed"),
            gen=msg.get("gen"),
        )
        return {"ok": True, "path": msg["out_path"], **info}
    if cmd == "design":
        info = model.design(
            text=msg["text"],
            instruct=msg["instruct"],
            out_path=msg["out_path"],
            seed=msg.get("seed"),
            gen=msg.get("gen"),
        )
        return {"ok": True, "path": msg["out_path"], **info}
    return {"ok": False, "error": f"unknown cmd {cmd!r}", "oom": False}


def _write(stream: TextIO, obj: dict[str, Any]) -> None:
    stream.write(json.dumps(obj) + "\n")
    stream.flush()


def serve(stdin: TextIO, stdout: TextIO, model: ModelHandle) -> None:
    """Read newline-JSON requests until EOF, writing one reply per request. Every exception
    becomes a structured ``{"ok": false, ...}`` reply (with an ``oom`` flag) so a single bad
    line never takes the worker down silently."""
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


def _claim_json_stdout() -> TextIO:
    """A private handle on the real stdout for JSON, with fd 1 itself pointed at stderr.

    Python-level prints are already diverted by redirect_stdout, but native code (and child
    processes such as a SoX probe) write straight to fd 1; after this they land on stderr."""
    json_fd = os.dup(sys.stdout.fileno())
    sys.stdout.flush()
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    return os.fdopen(json_fd, "w", encoding="utf-8", buffering=1)


def main(
    argv: list[str] | None = None,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> None:
    parser = argparse.ArgumentParser(description="Qwen3-TTS stdio worker")
    parser.add_argument("--models-dir", required=True, help="dir holding the Base/VoiceDesign dirs")
    parser.add_argument("--backend", choices=("fast", "stock"), default="fast")
    parser.add_argument("--language", default="English", help="text language (English, Auto, ...)")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="static KV cache (fast path)")
    args = parser.parse_args(argv)

    model = ModelHandle(
        args.models_dir,
        backend=args.backend,
        language=args.language,
        max_seq_len=args.max_seq_len,
    )
    out = stdout if stdout is not None else _claim_json_stdout()
    # Announce readiness on the pure-JSON channel (process booted; weights load lazily on `load`).
    _write(out, {"ok": True, "event": "ready", "protocol": PROTOCOL_VERSION})
    serve(stdin if stdin is not None else sys.stdin, out, model)


if __name__ == "__main__":
    main()
