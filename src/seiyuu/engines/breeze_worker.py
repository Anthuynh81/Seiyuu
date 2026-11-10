"""Breeze TTS 2 subprocess worker — runs in the SEPARATE Breeze env, NOT this venv.

Breeze TTS 2 (BreezeBlue, 3B; Qwen3-TTS codec lineage) is a source checkout, not a package: its
repo pins torch 2.9.1/cu128 + transformers 4.57.3, which conflict with this project's torch 2.6
(chatterbox's pin) and with the Qwen3-TTS env. So, like IndexTTS-2 and Qwen3-TTS, it runs
out-of-process here and ``breeze_engine.py`` (in the main venv) drives it over newline-delimited
JSON on stdin/stdout.

HARD RULE: this file is executed as a STANDALONE SCRIPT (``python breeze_worker.py``) by an
interpreter that has the Breeze dependencies and does NOT have seiyuu. So it must NEVER import
seiyuu, and it keeps module-top imports to the stdlib only — torch, numpy, soundfile and the
Breeze repo's own packages (``breeze_infer``, ``models``, imported from ``--repo-dir``) are
imported lazily, the first time the model loads.

One checkpoint, two request shapes, both cloning a voice from its reference clip AND the clip's
exact transcript (Breeze has no transcript-free clone mode):
- clone: no instruction, guidance 1.0 (the clone template defines no negative prompt, so any
  other scale is refused upstream).
- voice direction: the same clone plus a natural-language ``instruction`` ("calm, dry, a little
  lower"), classifier-free guidance at ``cfg_scale`` (default 4.0). Guidance doubles the batch.

Protocol (one JSON object per line, request -> reply):
- boot: the worker emits ``{"ok": true, "event": "ready", "protocol": N}`` once, unprompted.
- ``{"cmd": "ping"}``  -> ``{"ok": true, "protocol": N}``
- ``{"cmd": "load"}``  -> ``{"ok": true, "fast": "decode"|"off", "warmup_seconds": float}``
  Loads the checkpoint and, on the fast path, compiles + captures the graphs (16-50 s the first
  time; later starts reuse torch's inductor cache).
- ``{"cmd": "synthesize", "text", "reference_wav", "ref_text", "out_path", "reference_sha256"?,
     "seed"?, "instruction"?, "cfg_scale"?}``
  -> ``{"ok": true, "sample_rate": 24000, "path", "seconds", "frames", "hit_cap"}``
- any failure -> ``{"ok": false, "error": str, "oom": bool, "request_error": bool,
  "traceback": str}``. ``request_error`` marks a request that can never render as sent (the
  worker is healthy; the adapter fails the line without restarting it).
- EOF on stdin ends the loop (the adapter closes stdin to shut the worker down).

stdout MUST stay pure JSON. The Breeze repo prints progress lines from model construction and
graph capture, and torch.compile may spawn compilers; model calls run under
``redirect_stdout(sys.stderr)``, and when run as a script the worker also moves file descriptor 1
onto stderr (keeping a private dup for JSON), so even C-level or subprocess prints can't corrupt
the channel.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import traceback
from typing import Any, TextIO

# Bumped when the request/reply shape changes. The adapter refuses a worker that announces any
# other version at boot (SubprocessTransport(protocol=...)).
PROTOCOL_VERSION = 1

# The 12.5 Hz codec decodes to 24 kHz (1920 samples per frame). That is already Seiyuu's canonical
# rate, so the adapter's to_canonical() never resamples.
SAMPLE_RATE = 24000
FRAMES_PER_SECOND = 12.5

# Which stages of the Breeze runtime run compiled / as CUDA graphs. "decode" is the config
# measured on the 3080 Ti (2026-11-09): RTF 0.45 clone, 0.54 with cfg-4 direction, 8.8 GB peak.
# The text encoder and prefill stay eager on purpose: their graphs are frozen to the warmup
# profile's token buckets (a longer line raises), and the text encoder costs +1.5 GB for nothing.
FAST_CONFIGS: dict[str, dict[str, bool]] = {
    "decode": {"fast_depth_decoder": True, "fast_codec": True, "fast_backbone_decode": True},
    "off": {},
}

# Runtime constants, as upstream's infer.py and the A/B prototype ran them. MAX_SEQ_LEN is the
# static KV cache shared by the prompt (both transcripts + 12.5 reference frames per second) and
# the generated frames; it is also the model's position limit, so it is not a tunable.
MAX_SEQ_LEN = 2048
MAX_NEW_TOKENS = 1500  # 120 s: the runtime's own bound (and its per-request history buffer)
REPETITION_PENALTY = 1.1
# The sampler, for the backbone and the depth decoder alike. These equal what the checkout's
# update_generation_config_for_breeze applies at daa6cb2; they are pinned here (and passed to the
# runtime, where they win) so a later checkout can't change the sampling under an unchanged
# model_version.
TEMPERATURE = 0.9
TOP_K = 50
TOP_P = 1.0
ATTN_IMPLEMENTATION = "eager"  # the only one upstream exercises (flash_attn is not installed)
DEVICE = "cuda:0"
# Guidance for a line that carries an instruction and names no scale (upstream's recommendation).
DIRECTION_CFG_SCALE = 4.0

# Per-line generation cap. Measured on the 3080 Ti (2026-11-09): plain clones spoke 17.7-18.4
# characters/s and a directed line ("calm, dry and quiet") 16. 6 s plus one second per 6
# characters is about 3x that pace, loose on purpose (a directed line or a sigh runs slower);
# a runaway still stops long before the 120 s bound.
_CAP_FLOOR_SECONDS = 6.0
_CAP_CHARS_PER_SECOND = 6.0
_CAP_MIN_FRAMES = 64

# A line longer than SPLIT_OVER_CHARS is spoken in pieces: the ADAPTER splits it with
# split_long_text (below), sends one request per piece with the same reference, seed and
# direction, and joins the audio with PIECE_GAP_SECONDS of silence. Why 800: the prompt (the
# clip's frames, both transcripts, the line's own tokens) and the generated frames share
# MAX_SEQ_LEN positions. With the longest reference the clone flow binds (30 s = 375 frames,
# ~130 transcript tokens), 800 characters are ~200 text tokens and, at a slow 12 characters/s,
# ~835 frames: about 1,540 of 2,047, with room to spare. Whole paragraphs past that were cut
# short or refused. The longest line of the ghost-story book is 574 characters, so ordinary
# prose is never split. The gap: Breeze clips end in 0.18-0.30 s of quiet and start within
# 0.02 s (measured 2026-11-09), so 0.2 s more reads as the pause between two sentences.
SPLIT_OVER_CHARS = 800
PIECE_GAP_SECONDS = 0.2

# Everything above decides what a line sounds like or where it is cut, so it is part of the
# engine's model_version: bump GEN_RECIPE whenever max_frames_for, model_text, split_long_text
# or a constant in this block changes (FAST_CONFIGS entries, MAX_SEQ_LEN, MAX_NEW_TOKENS,
# REPETITION_PENALTY, the sampler, ATTN_IMPLEMENTATION, DIRECTION_CFG_SCALE, the cap constants,
# the split constants). Audio made under an old recipe is then never served from the cache under
# a new one. The Breeze checkout's own code (prompt templates, model code) is NOT part of the
# version: the README pins its revision.
# 2: long lines are split (recipe 1 cut them at the cap or refused them).
GEN_RECIPE = 2

# Breeze builds its prompt by joining text segments and re-tokenizing the joined string, so the
# tokenizer's special-token literals inside a book line, a transcript or an instruction would
# turn into control tokens: Breeze's own (a speaker switch, an instruction boundary, a
# reference-audio placeholder) and the Gemma tokenizer's (<eos>, <bos>, ...). None of them is
# speech; they are removed before the model sees the text.
_CONTROL_TOKENS = re.compile(
    r"<ins_bos>|<ins_eos>|<\|AUDIO\|>|<\|audio_eos\|>|\[S\d\]"
    r"|<(?:pad|eos|bos|unk|mask|start_of_turn|end_of_turn|start_of_image|end_of_image"
    r"|image_soft_token)>"
)


class RequestError(ValueError):
    """This request can never be rendered as sent (a prompt too long for the model, a missing
    transcript, a bad guidance scale). Reported with ``request_error: true`` so the adapter fails
    the line without restarting a healthy worker: a fresh process would refuse it the same way."""


def model_text(text: str) -> str:
    """The text actually sent to the model: the tokenizer's control-token literals removed."""
    return _CONTROL_TOKENS.sub(" ", text).strip() if _CONTROL_TOKENS.search(text) else text


# A sentence end: closing punctuation, any closing quotes or brackets, then whitespace.
_SENTENCE_END = re.compile(r"[.!?][\"'”’)\]]*\s+")
_CLAUSE_END = re.compile(r"[,;:][\"'”’)\]]*\s+")
_SPACE = re.compile(r"\s+")


def _cut_points(text: str, pattern: re.Pattern[str], *, sentences: bool) -> list[int]:
    """Offsets where ``text`` may be cut: the start of the unit that follows each match."""
    points = []
    for match in pattern.finditer(text):
        end = match.end()
        if end >= len(text):
            continue
        if sentences:
            # `“Stop!” he cried.` is one sentence: a lower-case continuation is no boundary.
            if text[end].islower():
                continue
            # an initial ("J. R. R. Tolkien", "U.S. Army"), not the end of a sentence
            dot = match.start()
            if (
                text[dot] == "."
                and dot >= 1
                and text[dot - 1].isupper()
                and (dot == 1 or not text[dot - 2].isalpha())
            ):
                continue
        points.append(end)
    return points


def _units(text: str, limit: int) -> list[str]:
    """``text`` as units no longer than ``limit``, cut at the best boundary that allows it:
    sentence ends first, then clause punctuation, then spaces, then (a run with no space at
    all) wherever the limit falls."""
    text = text.strip()
    if len(text) <= limit:
        return [text] if text else []
    for pattern, sentences in ((_SENTENCE_END, True), (_CLAUSE_END, False), (_SPACE, False)):
        points = _cut_points(text, pattern, sentences=sentences)
        if points:
            edges = [0, *points, len(text)]
            out: list[str] = []
            for start, end in zip(edges, edges[1:], strict=False):
                out.extend(_units(text[start:end], limit))
            return out
    return [text[i : i + limit] for i in range(0, len(text), limit)]


def _has_words(text: str) -> bool:
    return any(ch.isalnum() for ch in text)


def _fill(units: list[str], limit: int) -> list[str]:
    """Pack units into pieces, each filled as far as ``limit`` allows."""
    pieces: list[str] = []
    current = ""
    for unit in units:
        joined = f"{current} {unit}" if current else unit
        if current and len(joined) > limit:
            pieces.append(current)
            current = unit
        else:
            current = joined
    if current:
        pieces.append(current)
    return pieces


def split_long_text(text: str, limit: int = SPLIT_OVER_CHARS) -> list[str]:
    """The pieces a line is spoken in. A line of at most ``limit`` characters is returned as it
    is, in one piece. A longer one is cut at sentence ends (see :func:`_units` for a sentence
    that is itself too long) and packed into the fewest pieces that fit, of even length, none
    over ``limit``. Every word is kept, in order; only the whitespace at a cut is dropped. A
    unit with nothing to say (no letter or digit) rides with its neighbour, so no request is
    ever made for a stray quote mark."""
    if len(text) <= limit:
        return [text]
    units: list[str] = []
    for unit in _units(text, limit):
        if units and not (_has_words(unit) and _has_words(units[-1])):
            units[-1] = f"{units[-1]} {unit}"
        else:
            units.append(unit)
    count = len(_fill(units, limit))  # the fewest pieces these units allow
    # Same count, evened out: each piece aims at an equal share of what is left, so a paragraph
    # does not end in one short orphan sentence spoken by itself.
    pieces: list[str] = []
    i = 0
    for slots in range(count, 0, -1):
        left = units[i:]
        if not left:
            break
        if slots == 1:
            pieces.extend(_fill(left, limit))  # whatever is left (more than one only if uneven)
            break
        share = (sum(len(u) for u in left) + len(left) - 1) / slots
        current = left[0]
        i += 1
        while i < len(units):
            joined = f"{current} {units[i]}"
            if len(joined) > limit or abs(len(joined) - share) > abs(len(current) - share):
                break
            current = joined
            i += 1
        pieces.append(current)
    return pieces


def max_frames_for(text: str, prompt_len: int) -> int:
    """Generation cap for one line, in codec frames.

    ``prompt_len`` is the prompt's token length (both transcripts, the reference frames, the
    instruction). The runtime stops SILENTLY when prompt + generation reaches MAX_SEQ_LEN, so the
    cap is bounded by the room left: the cap stays the binding stop and ``hit_cap`` stays
    truthful. Raises when the prompt leaves no usable room (a very long line or reference)."""
    room = MAX_SEQ_LEN - 1 - prompt_len
    if room < _CAP_MIN_FRAMES:
        raise RequestError(
            f"prompt too long: {prompt_len} tokens of reference clip, transcripts and text "
            f"leave {max(room, 0)} of {MAX_SEQ_LEN} positions to speak in (use a shorter "
            "reference clip or split the line)"
        )
    seconds = _CAP_FLOOR_SECONDS + len(text) / _CAP_CHARS_PER_SECOND
    frames = math.ceil(seconds * FRAMES_PER_SECOND)
    return max(_CAP_MIN_FRAMES, min(frames, MAX_NEW_TOKENS, room))


def guidance_scale_for(instruction: str, cfg_scale: Any) -> float:
    """Classifier-free guidance for one request. Without an instruction it is always 1.0 (the
    clone template has no negative prompt); with one, the request's scale or the default."""
    if not instruction:
        return 1.0
    scale = DIRECTION_CFG_SCALE if cfg_scale is None else float(cfg_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise RequestError(f"cfg_scale must be a finite number above 0, got {cfg_scale!r}")
    return scale


def compile_env(*, windows: bool | None = None) -> None:
    """Environment torch must see BEFORE it is imported (it reads these at import time).

    On Windows inductor's static CUDA launcher fails the first compiled kernel with
    ``OverflowError: Python int too large to convert to C long``; the dynamic launcher works
    (with triton-windows installed). Elsewhere the default is left alone."""
    on_windows = os.name == "nt" if windows is None else windows
    if on_windows:
        os.environ["TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER"] = "0"


def _is_oom(exc: BaseException) -> bool:
    """True for a CUDA out-of-memory error, however torch spells it across versions."""
    return "OutOfMemory" in type(exc).__name__ or "out of memory" in str(exc).lower()


class _CapReached(Exception):
    """Raised from the per-frame observer to stop a line that ran past its cap."""


class ModelHandle:
    """Lazily loads the Breeze checkpoint and runs inference. torch and the Breeze repo imports
    live here so the worker process can boot (and answer ``ping``) before paying the load."""

    def __init__(self, repo_dir: str, checkpoint_dir: str, *, fast: str = "decode") -> None:
        if fast not in FAST_CONFIGS:
            raise ValueError(
                f"unknown fast config {fast!r}; expected one of {sorted(FAST_CONFIGS)}"
            )
        self._repo_dir = repo_dir
        self._checkpoint_dir = checkpoint_dir
        self.fast = fast
        self._runtime: Any | None = None
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._audio_tokenizer: Any | None = None
        self._templates: Any | None = None  # the breeze_infer.templates module
        self._set_all_seeds: Any | None = None
        self.warmup_seconds = 0.0
        self._requests = 0
        # Reference clips encoded so far: content hash (or path) -> codec codes on the CPU.
        # Upstream re-encodes the clip on every line, twice with guidance; the codes are a few KB
        # per voice. Keyed by CONTENT, so a re-cloned voice (same path, new audio) is re-encoded.
        self._reference_codes: dict[str, Any] = {}
        self._reference_key: str | None = None

    def load(self) -> None:
        if self._runtime is not None:
            return
        compile_env()  # before the first torch import, or the Windows fix is read too late
        from contextlib import redirect_stdout

        with redirect_stdout(sys.stderr):
            from dataclasses import replace
            from pathlib import Path

            if self._repo_dir not in sys.path:
                # The repo is a source checkout exposing two top-level packages, `breeze_infer`
                # and the generically named `models`: it goes first so nothing shadows them.
                sys.path.insert(0, self._repo_dir)
            from breeze_infer import templates
            from breeze_infer.runtime import (
                load_runtime,
                set_all_seeds,
                update_generation_config_for_breeze,
            )
            from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig

            tokenizer, model, audio_tokenizer = load_runtime(
                Path(self._checkpoint_dir),
                device=DEVICE,
                attn_implementation=ATTN_IMPLEMENTATION,
            )
            update_generation_config_for_breeze(model)
            config = FastStreamingConfig(
                max_new_tokens=MAX_NEW_TOKENS,
                max_seq_len=MAX_SEQ_LEN,
                repetition_penalty=REPETITION_PENALTY,
                temperature=TEMPERATURE,
                top_k=TOP_K,
                top_p=TOP_P,
                do_sample=True,
                **FAST_CONFIGS[self.fast],
            )
            runtime = FastBreezeStreamingRuntime(
                model, audio_tokenizer, config, tokenizer=tokenizer
            )
            started = time.perf_counter()
            if runtime.fast_enabled:
                # Compile + capture now: inside the load timeout rather than the first line's,
                # and never between that line's seeding and its sampling. The stock profile warms
                # both batch shapes (guidance off and on), so clone and direction lines can
                # alternate without recapturing.
                from models.warmup_profile import load_warmup_profile

                profile = load_warmup_profile(Path(self._repo_dir) / "configs" / "fast.json")
                runtime.warmup_from_profile(
                    replace(profile, codec_chunk_frames=runtime.codec_chunk_frames)
                )
            self.warmup_seconds = round(time.perf_counter() - started, 1)
            self._install_reference_cache(templates)
            self._templates = templates
            self._set_all_seeds = set_all_seeds
            self._tokenizer, self._model, self._audio_tokenizer = tokenizer, model, audio_tokenizer
            self._runtime = runtime

    def _install_reference_cache(self, templates: Any) -> None:
        """Wrap the repo's one reference-encoding seam so each clip is encoded once per worker.

        ``prepare_inputs`` reaches the codec only through ``templates._encode_prompt_audio``; the
        returned codes are never mutated downstream (every use copies), so sharing them is safe."""
        encode = templates._encode_prompt_audio

        def cached(audio_tokenizer: Any, audio_path: str) -> Any:
            key = self._reference_key or audio_path
            codes = self._reference_codes.get(key)
            if codes is None:
                codes = encode(audio_tokenizer, audio_path)
                self._reference_codes[key] = codes
            return codes

        templates._encode_prompt_audio = cached

    def synthesize(
        self,
        *,
        text: str,
        reference_wav: str,
        ref_text: str,
        out_path: str,
        reference_sha256: str | None = None,
        seed: int | None = None,
        instruction: str | None = None,
        cfg_scale: float | None = None,
    ) -> dict[str, Any]:
        from contextlib import redirect_stdout

        ref_text = (ref_text or "").strip()
        if not ref_text:
            raise RequestError(
                "breeze needs the reference clip's exact transcript (ref_text); it has no "
                "transcript-free clone mode"
            )
        if not text.strip():
            raise RequestError("no text left to speak once control-token literals are removed")
        instruction = (instruction or "").strip()
        scale = guidance_scale_for(instruction, cfg_scale)
        self.load()
        self._requests += 1
        request_id = f"seiyuu-{self._requests}"
        request: dict[str, Any] = {
            "id": request_id,
            "text": text,
            "speaker": "S0",
            "ref_audio_path": reference_wav,
            "ref_text": ref_text,
        }
        if instruction:
            request["instruction"] = instruction
        templates = self._templates
        steps = 0
        chunks: list[Any] = []
        with redirect_stdout(sys.stderr):
            if seed is not None:
                # random + numpy + torch, as upstream seeds a request. numpy refuses a seed
                # outside 32 bits, hence the fold; the audio is decided by the runtime's own torch
                # reseed right before sampling (after its lazy setup), from the full value.
                self._set_all_seeds(int(seed) % 2**32)  # type: ignore[misc]
            self._reference_key = reference_sha256 or reference_wav
            try:
                inputs = templates.prepare_inputs(  # type: ignore[union-attr]
                    self._tokenizer,
                    self._audio_tokenizer,
                    self._model,
                    [request],
                    templates.get_template(templates.select_template_name(request)),  # type: ignore[union-attr]
                    guidance_scale=scale,
                    guidance_scale_ref=None,  # dual guidance is refused by the streaming runtime
                    guidance_scale_ins=None,
                )
            finally:
                self._reference_key = None
            # With guidance the two prompts are padded to the longer one.
            prompt_len = int(inputs["input_ids"].shape[1])
            negative = inputs.get("cfg_negative_prompt_ids")
            if negative is not None:
                prompt_len = max(prompt_len, int(negative.shape[1]))
            cap = max_frames_for(text, prompt_len)
            # The eager codec decodes two frames at a time: a cap that lands mid-chunk would
            # drop the buffered frame. (One frame per chunk on the fast path: a no-op there.)
            cap -= cap % int(self._runtime.codec_chunk_frames)  # type: ignore[union-attr]
            # Where the runtime itself stops without a word: its token bound, or a full cache.
            silent_stop = min(MAX_NEW_TOKENS, MAX_SEQ_LEN - 1 - prompt_len)
            capped = False

            def observe(_frame: Any) -> None:
                # Called once per generated frame, before it is decoded. One frame past the cap
                # means the model never emitted its end token: stop it here.
                nonlocal steps
                steps += 1
                if steps > cap:
                    raise _CapReached

            stream = self._runtime.iter_audio_chunks(  # type: ignore[union-attr]
                inputs,
                request_id=request_id,
                seed=None if seed is None else int(seed),
                token_observer=observe,
            )
            try:
                for chunk in stream:
                    chunks.append(chunk.audio)
            except _CapReached:
                capped = True  # the generator's own cleanup closed the codec request
            finally:
                stream.close()
        # A line that ends on its own exactly at the cap is not a runaway; one the runtime
        # stopped silently is.
        return _write_wav(out_path, chunks, min(steps, cap), capped or steps >= silent_stop)

    def empty_cache(self) -> None:
        """Best-effort VRAM reclaim after an OOM (the adapter restarts the worker anyway: the
        runtime has no unload path and a failed graph capture can leave it unusable)."""
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 — cleanup must never mask the original error
            pass


def _write_wav(out_path: str, chunks: list[Any], frames: int, hit_cap: bool) -> dict[str, Any]:
    """Write the generated chunks as 16-bit mono PCM and describe the line for the reply.

    A generation that produced no audio (the first token was the end token) is written as an
    EMPTY file so the adapter fails it loudly instead of shipping silence as speech."""
    import numpy as np
    import soundfile as sf

    parts = [np.asarray(c, dtype=np.float32).reshape(-1) for c in chunks]
    samples = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    sf.write(out_path, pcm, SAMPLE_RATE, subtype="PCM_16")
    return {
        "sample_rate": SAMPLE_RATE,
        "seconds": round(samples.size / float(SAMPLE_RATE), 3),
        "frames": frames,
        # Reached the cap: the model most likely never emitted its end token (a runaway), or the
        # prompt left too little room for the line.
        "hit_cap": hit_cap,
    }


def handle(msg: dict[str, Any], model: ModelHandle) -> dict[str, Any]:
    """Dispatch one decoded request to a reply dict. Pure of I/O so it is unit-testable
    with a fake model; ``serve`` owns the stdin/stdout framing and error wrapping."""
    cmd = msg.get("cmd")
    if cmd == "ping":
        return {"ok": True, "protocol": PROTOCOL_VERSION}
    if cmd == "load":
        model.load()
        return {"ok": True, "fast": model.fast, "warmup_seconds": model.warmup_seconds}
    if cmd == "synthesize":
        info = model.synthesize(
            text=model_text(msg["text"]),
            reference_wav=msg["reference_wav"],
            ref_text=model_text(msg.get("ref_text") or ""),
            out_path=msg["out_path"],
            reference_sha256=msg.get("reference_sha256"),
            seed=msg.get("seed"),
            instruction=model_text(msg.get("instruction") or "") or None,
            cfg_scale=msg.get("cfg_scale"),
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
                "request_error": isinstance(exc, RequestError),
                "traceback": traceback.format_exc(),
            }
        _write(stdout, reply)


def _claim_json_stdout() -> TextIO:
    """A private handle on the real stdout for JSON, with fd 1 itself pointed at stderr.

    Python-level prints are already diverted by redirect_stdout, but native code and child
    processes (inductor's compiler runs) write straight to fd 1; after this they land on stderr."""
    json_fd = os.dup(sys.stdout.fileno())
    sys.stdout.flush()
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    return os.fdopen(json_fd, "w", encoding="utf-8", buffering=1)


def _drop_script_dir() -> None:
    """Take this file's own directory off ``sys.path`` (Python puts a script's directory first).

    It holds Seiyuu's other engine modules (``audio.py``, ``base.py``), which the worker never
    imports and which must not shadow a top-level module the Breeze stack asks for."""
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != here]


def main(
    argv: list[str] | None = None,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> None:
    parser = argparse.ArgumentParser(description="Breeze TTS 2 stdio worker")
    parser.add_argument("--repo-dir", required=True, help="the breeze-tts source checkout")
    parser.add_argument("--checkpoint-dir", required=True, help="the breeze-tts-2 weights dir")
    parser.add_argument("--fast", choices=sorted(FAST_CONFIGS), default="decode")
    args = parser.parse_args(argv)

    model = ModelHandle(args.repo_dir, args.checkpoint_dir, fast=args.fast)
    out = stdout if stdout is not None else _claim_json_stdout()
    # Announce readiness on the pure-JSON channel (process booted; weights load lazily on `load`).
    _write(out, {"ok": True, "event": "ready", "protocol": PROTOCOL_VERSION})
    serve(stdin if stdin is not None else sys.stdin, out, model)


if __name__ == "__main__":
    _drop_script_dir()
    main()
