"""Breeze TTS 2 worker tests — no Breeze repo, no GPU.

The request dispatch (handle/serve) runs against a fake model, and ``ModelHandle`` runs against
fake ``breeze_infer`` / ``models`` packages, so the call shapes (clone vs direction, guidance,
the per-line seed, the encoded-reference cache, the generation cap) are pinned without the real
checkout. The real load/generate path only runs in the separate env (tests/test_breeze_gpu.py).
"""

import io
import json
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from seiyuu.engines import breeze_worker as w

# -- pure helpers -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('"Hello there."', '"Hello there."'),  # quotes stay: Breeze was chosen on quoted lines
        ("(sigh) Fine.", "(sigh) Fine."),  # a performed sound, not a control token
        ("[S1] Hello.", "Hello."),  # a literal speaker tag would switch speakers
        ("Go <ins_bos>shout<ins_eos> now", "Go  shout  now"),
        ("<|AUDIO|><|audio_eos|>Hi", "Hi"),
        ("[Sx] and [S12] are just text", "[Sx] and [S12] are just text"),
    ],
)
def test_model_text_removes_control_token_literals(text, expected):
    assert w.model_text(text) == expected


def test_max_frames_scales_with_text_and_respects_the_prompt():
    assert w.max_frames_for("Yes.", 200) == int(np.ceil((6 + 4 / 6) * 12.5))  # 84 frames
    assert w.max_frames_for("", 200) == 75  # the floor alone, above the 64-frame minimum
    assert w.max_frames_for("x" * 300, 200) == int(np.ceil((6 + 300 / 6) * 12.5))
    assert w.max_frames_for("x" * 9000, 200) == w.MAX_NEW_TOKENS  # the runtime's own bound
    # the prompt shares the 2048-position cache: a long reference leaves less room to speak
    assert w.max_frames_for("x" * 9000, 1500) == 2048 - 1 - 1500
    assert w.max_frames_for("x" * 300, 2048 - 1 - 64) == 64
    with pytest.raises(ValueError, match="prompt too long"):
        w.max_frames_for("Hi.", 2048 - 64)
    with pytest.raises(ValueError, match="leave 0 of 2048"):
        w.max_frames_for("Hi.", 5000)


def test_guidance_is_off_without_an_instruction():
    assert w.guidance_scale_for("", None) == 1.0
    assert w.guidance_scale_for("", 4.0) == 1.0  # the clone template has no negative prompt
    assert w.guidance_scale_for("calm", None) == w.DIRECTION_CFG_SCALE == 4.0
    assert w.guidance_scale_for("calm", 2) == 2.0
    for bad in (0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="cfg_scale"):
            w.guidance_scale_for("calm", bad)


def test_gen_recipe_is_bumped_with_the_recipe():
    # Any of these changes what a line sounds like or where it is cut. If this fails, bump
    # GEN_RECIPE in breeze_worker.py (it is part of model_version) and update the pins here.
    pinned = (
        w.FAST_CONFIGS,
        (w.MAX_SEQ_LEN, w.MAX_NEW_TOKENS, w.REPETITION_PENALTY, w.ATTN_IMPLEMENTATION),
        w.DIRECTION_CFG_SCALE,
        (w._CAP_FLOOR_SECONDS, w._CAP_CHARS_PER_SECOND, w._CAP_MIN_FRAMES),
        w.max_frames_for("x" * 473, 300),
        w.model_text("[S0]<ins_bos>a<ins_eos>"),
    )
    assert (w.GEN_RECIPE, pinned) == (
        1,
        (
            {
                "decode": {
                    "fast_depth_decoder": True,
                    "fast_codec": True,
                    "fast_backbone_decode": True,
                },
                "off": {},
            },
            (2048, 1500, 1.1, "eager"),
            4.0,
            (6.0, 6.0, 64),
            1061,
            "a",
        ),
    )


def test_compile_env_is_a_windows_fix(monkeypatch):
    monkeypatch.delenv("TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER", raising=False)
    w.compile_env(windows=False)
    assert "TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER" not in os.environ
    monkeypatch.setenv("TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER", "1")  # inherited from a shell
    w.compile_env(windows=True)
    assert os.environ["TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER"] == "0"


def test_write_wav_is_int16_mono_24k_and_flags_the_cap(tmp_path):
    out = tmp_path / "o.wav"
    chunks = [np.full(12000, 0.5, dtype=np.float32), np.full((1, 12000), 2.0, dtype=np.float32)]
    info = w._write_wav(str(out), chunks, 13, 100)
    data, sr = sf.read(out, dtype="int16")
    assert sr == 24000 and data.ndim == 1 and data.size == 24000
    assert sf.info(str(out)).subtype == "PCM_16"
    assert data.max() == 32767  # clipped, never wrapped
    assert info == {"sample_rate": 24000, "seconds": 1.0, "frames": 13, "hit_cap": False}
    assert w._write_wav(str(out), chunks, 100, 100)["hit_cap"] is True


def test_no_audio_writes_an_empty_file(tmp_path):
    out = tmp_path / "o.wav"
    info = w._write_wav(str(out), [], 0, 100)
    assert sf.read(out)[0].size == 0  # the adapter fails an empty file loudly
    assert info["seconds"] == 0.0 and info["hit_cap"] is False


def test_the_script_dir_is_dropped_from_sys_path(monkeypatch):
    engines_dir = str(Path(w.__file__).resolve().parent)
    monkeypatch.setattr(sys, "path", [engines_dir, "/somewhere/else", engines_dir + os.sep])
    w._drop_script_dir()
    assert sys.path == ["/somewhere/else"]


# -- dispatch ---------------------------------------------------------------------------------


class _FakeModel:
    fast = "decode"
    warmup_seconds = 21.5

    def __init__(self, *, oom=False):
        self.calls = []
        self.loads = 0
        self.emptied = 0
        self._oom = oom

    def load(self):
        self.loads += 1

    def synthesize(self, **kw):
        self.calls.append(kw)
        if self._oom:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return {"sample_rate": 24000, "seconds": 1.0, "frames": 13, "hit_cap": False}

    def empty_cache(self):
        self.emptied += 1


def test_ping_and_load():
    model = _FakeModel()
    assert w.handle({"cmd": "ping"}, model) == {"ok": True, "protocol": w.PROTOCOL_VERSION}
    assert w.handle({"cmd": "load"}, model) == {
        "ok": True,
        "fast": "decode",
        "warmup_seconds": 21.5,
    }
    assert model.loads == 1


def test_synthesize_forwards_everything_and_cleans_the_texts():
    model = _FakeModel()
    reply = w.handle(
        {
            "cmd": "synthesize",
            "text": '"Hi." [S2]',
            "reference_wav": "/v/x/reference.wav",
            "reference_sha256": "abc",
            "ref_text": "I keep a notebook.<ins_eos>",
            "out_path": "/tmp/o.wav",
            "seed": 7,
            "instruction": "calm<ins_bos>",
            "cfg_scale": 3.0,
        },
        model,
    )
    assert reply == {
        "ok": True,
        "path": "/tmp/o.wav",
        "sample_rate": 24000,
        "seconds": 1.0,
        "frames": 13,
        "hit_cap": False,
    }
    assert model.calls == [
        {
            "text": '"Hi."',
            "reference_wav": "/v/x/reference.wav",
            "ref_text": "I keep a notebook.",
            "out_path": "/tmp/o.wav",
            "reference_sha256": "abc",
            "seed": 7,
            "instruction": "calm",
            "cfg_scale": 3.0,
        }
    ]


def test_synthesize_defaults_for_a_plain_clone():
    model = _FakeModel()
    request = {"text": "Hi.", "reference_wav": "r", "ref_text": "T.", "out_path": "o"}
    w.handle({"cmd": "synthesize", **request}, model)
    call = model.calls[0]
    assert call["instruction"] is None and call["cfg_scale"] is None
    assert call["seed"] is None and call["reference_sha256"] is None


def test_unknown_cmd():
    assert w.handle({"cmd": "design"}, _FakeModel())["ok"] is False


def test_serve_reports_oom_and_keeps_going():
    model = _FakeModel(oom=True)
    stdin = io.StringIO(
        json.dumps({"cmd": "synthesize", "text": "a", "reference_wav": "r", "out_path": "o"})
        + "\nnot json\n\n"
        + json.dumps({"cmd": "ping"})
        + "\n"
    )
    stdout = io.StringIO()
    w.serve(stdin, stdout, model)
    replies = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert replies[0]["ok"] is False and replies[0]["oom"] is True and replies[0]["traceback"]
    assert model.emptied == 1
    assert replies[1]["ok"] is False and "bad json" in replies[1]["error"]
    assert replies[2] == {"ok": True, "protocol": w.PROTOCOL_VERSION}


def test_a_bad_request_is_an_error_reply_not_a_crash():
    stdout = io.StringIO()
    w.serve(io.StringIO(json.dumps({"cmd": "synthesize"}) + "\n"), stdout, _FakeModel())
    reply = json.loads(stdout.getvalue())
    assert reply["ok"] is False and reply["oom"] is False and "KeyError" in reply["error"]


def test_main_announces_ready_and_exits_at_eof():
    out = io.StringIO()
    w.main(
        ["--repo-dir", "/r", "--checkpoint-dir", "/c", "--fast", "off"],
        stdin=io.StringIO(""),
        stdout=out,
    )
    assert json.loads(out.getvalue()) == {
        "ok": True,
        "event": "ready",
        "protocol": w.PROTOCOL_VERSION,
    }
    with pytest.raises(SystemExit):
        w.main(["--repo-dir", "/r", "--checkpoint-dir", "/c", "--fast", "all"], stdout=out)


# -- ModelHandle against a fake Breeze checkout -----------------------------------------------


class _Ids:
    def __init__(self, length):
        self.shape = (1, length)


@dataclass(frozen=True)
class _Profile:
    path: str
    codec_chunk_frames: int = 2


@dataclass
class _Chunk:
    audio: np.ndarray


class _Config:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _Runtime:
    """Stands in for models.fast_streaming.FastBreezeStreamingRuntime."""

    def __init__(self, model, audio_tokenizer, config, *, tokenizer):
        self.model, self.audio_tokenizer, self.tokenizer = model, audio_tokenizer, tokenizer
        self.config = config
        self.fast_enabled = any(v for k, v in config.kwargs.items() if k.startswith("fast_"))
        self.codec_chunk_frames = 1 if config.kwargs.get("fast_codec") else 2
        self.warmed = []
        self.calls = []
        self.closed = []
        self.frames_wanted = 5  # how many frames the "model" speaks before its end token

    def warmup_from_profile(self, profile):
        self.warmed.append(profile)

    def iter_audio_chunks(self, inputs, *, request_id, seed, token_observer):
        self.calls.append({"inputs": inputs, "request_id": request_id, "seed": seed})
        try:
            for i in range(self.frames_wanted):
                token_observer(i)
                yield _Chunk(np.full(1920, 0.1, dtype=np.float32))
        finally:
            self.closed.append(request_id)


@pytest.fixture
def breeze(monkeypatch):
    """Fake ``breeze_infer`` and ``models`` packages; returns a log of what the worker did."""
    log = types.SimpleNamespace(
        events=[], loads=[], seeds=[], prepared=[], encoded=[], runtimes=[], prompt_len=200
    )
    monkeypatch.setattr(sys, "path", list(sys.path))  # the worker prepends the repo dir
    monkeypatch.delenv("TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER", raising=False)
    monkeypatch.setattr(w, "compile_env", lambda: log.events.append("compile_env"))

    runtime_mod = types.ModuleType("breeze_infer.runtime")

    def load_runtime(ckpt_dir, *, device, attn_implementation):
        log.events.append("load_runtime")
        log.loads.append((ckpt_dir, device, attn_implementation))
        return "tokenizer", "model", "audio_tokenizer"

    runtime_mod.load_runtime = load_runtime
    runtime_mod.set_all_seeds = log.seeds.append
    runtime_mod.update_generation_config_for_breeze = lambda model: log.events.append("genconfig")

    templates_mod = types.ModuleType("breeze_infer.templates")

    def _encode_prompt_audio(audio_tokenizer, audio_path):
        log.encoded.append(audio_path)
        return f"codes:{audio_path}:{len(log.encoded)}"

    def select_template_name(request):
        return "ref_edit_tata" if request.get("instruction") else "ref_clone_tata"

    def prepare_inputs(tokenizer, audio_tokenizer, model, requests, template, **guidance):
        (request,) = requests
        # as upstream: the reference is encoded once per prompt, and guidance builds a second
        # (negative) prompt around the same clip
        branches = 1 if guidance["guidance_scale"] == 1.0 else 2
        codes = [
            templates_mod._encode_prompt_audio(audio_tokenizer, request["ref_audio_path"])
            for _ in range(branches)
        ]
        log.prepared.append(
            {"request": dict(request), "template": template, "codes": codes, **guidance}
        )
        inputs = {"input_ids": _Ids(log.prompt_len)}
        if branches == 2:
            inputs["cfg_negative_prompt_ids"] = _Ids(log.prompt_len + 7)
        return inputs

    templates_mod._encode_prompt_audio = _encode_prompt_audio
    templates_mod.select_template_name = select_template_name
    templates_mod.get_template = lambda name: f"template:{name}"
    templates_mod.prepare_inputs = prepare_inputs

    fast_mod = types.ModuleType("models.fast_streaming")
    fast_mod.FastStreamingConfig = _Config

    def build_runtime(*args, **kwargs):
        log.runtimes.append(_Runtime(*args, **kwargs))
        return log.runtimes[-1]

    fast_mod.FastBreezeStreamingRuntime = build_runtime
    warm_mod = types.ModuleType("models.warmup_profile")
    warm_mod.load_warmup_profile = lambda path: _Profile(str(path))

    infer_pkg = types.ModuleType("breeze_infer")
    infer_pkg.templates, infer_pkg.runtime = templates_mod, runtime_mod
    models_pkg = types.ModuleType("models")
    for name, mod in {
        "breeze_infer": infer_pkg,
        "breeze_infer.runtime": runtime_mod,
        "breeze_infer.templates": templates_mod,
        "models": models_pkg,
        "models.fast_streaming": fast_mod,
        "models.warmup_profile": warm_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return log


def _handle(tmp_path, **kw):
    return w.ModelHandle(str(tmp_path / "repo"), str(tmp_path / "ckpt"), **kw)


def _synth(handle, tmp_path, **kw):
    params = dict(
        text="Hello there.",
        reference_wav=str(tmp_path / "reference.wav"),
        ref_text="I keep a notebook.",
        out_path=str(tmp_path / "o.wav"),
        reference_sha256="sha1",
        seed=1234,
    )
    params.update(kw)
    return handle.synthesize(**params)


def test_load_builds_the_decode_runtime_and_warms_it(breeze, tmp_path):
    handle = _handle(tmp_path)
    handle.load()
    # the Windows compile fix lands before anything that could import torch
    assert breeze.events == ["compile_env", "load_runtime", "genconfig"]
    assert breeze.loads == [(tmp_path / "ckpt", "cuda:0", "eager")]
    assert sys.path[0] == str(tmp_path / "repo")
    (runtime,) = breeze.runtimes
    assert runtime.config.kwargs == {
        "max_new_tokens": 1500,
        "max_seq_len": 2048,
        "repetition_penalty": 1.1,
        "fast_depth_decoder": True,
        "fast_codec": True,
        "fast_backbone_decode": True,
    }
    assert (runtime.model, runtime.audio_tokenizer, runtime.tokenizer) == (
        "model",
        "audio_tokenizer",
        "tokenizer",
    )
    # graphs are captured at load, with the profile matched to this runtime's codec chunking
    assert runtime.warmed == [
        _Profile(str(tmp_path / "repo" / "configs" / "fast.json"), codec_chunk_frames=1)
    ]
    handle.load()
    assert len(breeze.runtimes) == 1 and sys.path.count(str(tmp_path / "repo")) == 1


def test_fast_off_runs_eager_and_skips_the_warmup(breeze, tmp_path):
    handle = _handle(tmp_path, fast="off")
    handle.load()
    (runtime,) = breeze.runtimes
    assert not any(k.startswith("fast_") for k in runtime.config.kwargs)
    assert runtime.warmed == [] and handle.fast == "off"
    with pytest.raises(ValueError, match="unknown fast config"):
        _handle(tmp_path, fast="all")


def test_clone_request_shape(breeze, tmp_path):
    handle = _handle(tmp_path)
    info = _synth(handle, tmp_path, ref_text="  I keep a notebook.  ")
    (prepared,) = breeze.prepared
    assert prepared["request"] == {
        "id": "seiyuu-1",
        "text": "Hello there.",
        "speaker": "S0",
        "ref_audio_path": str(tmp_path / "reference.wav"),
        "ref_text": "I keep a notebook.",
    }
    assert prepared["template"] == "template:ref_clone_tata"
    assert prepared["guidance_scale"] == 1.0
    assert prepared["guidance_scale_ref"] is None and prepared["guidance_scale_ins"] is None
    (runtime,) = breeze.runtimes
    assert breeze.seeds == [1234]  # random/numpy/torch, before the prompt is built
    assert runtime.calls[0]["seed"] == 1234 and runtime.calls[0]["request_id"] == "seiyuu-1"
    assert runtime.closed == ["seiyuu-1"]
    assert info == {"sample_rate": 24000, "seconds": 0.4, "frames": 5, "hit_cap": False}
    assert sf.read(tmp_path / "o.wav")[0].size == 5 * 1920
    _synth(handle, tmp_path, seed=None)
    assert breeze.seeds == [1234] and runtime.calls[1]["seed"] is None
    assert runtime.calls[1]["request_id"] == "seiyuu-2"


def test_direction_adds_the_instruction_and_guidance(breeze, tmp_path):
    handle = _handle(tmp_path)
    _synth(handle, tmp_path, instruction="  calm and dry ", cfg_scale=None)
    _synth(handle, tmp_path, instruction="calm and dry", cfg_scale=2.5)
    _synth(handle, tmp_path, instruction="   ", cfg_scale=2.5)  # blank: a plain clone
    first, second, third = breeze.prepared
    assert first["request"]["instruction"] == "calm and dry"
    assert first["template"] == "template:ref_edit_tata" and first["guidance_scale"] == 4.0
    assert second["guidance_scale"] == 2.5
    assert "instruction" not in third["request"] and third["guidance_scale"] == 1.0
    with pytest.raises(ValueError, match="cfg_scale"):
        _synth(handle, tmp_path, instruction="calm", cfg_scale=0)


def test_the_reference_is_encoded_once_per_clip(breeze, tmp_path):
    handle = _handle(tmp_path)
    _synth(handle, tmp_path)
    _synth(handle, tmp_path, instruction="calm")  # guidance: two prompts, same clip
    _synth(handle, tmp_path, text="Another line.")
    ref = str(tmp_path / "reference.wav")
    assert breeze.encoded == [ref]
    assert {c for p in breeze.prepared for c in p["codes"]} == {f"codes:{ref}:1"}
    # same path, new audio (a re-clone): a new content hash re-encodes
    _synth(handle, tmp_path, reference_sha256="sha2")
    assert breeze.encoded == [ref, ref]
    # without a hash the path is the key
    _synth(handle, tmp_path, reference_sha256=None)
    _synth(handle, tmp_path, reference_sha256=None)
    assert len(breeze.encoded) == 3 and handle._reference_key is None


def test_a_runaway_is_cut_at_the_cap(breeze, tmp_path):
    handle = _handle(tmp_path)
    handle.load()
    (runtime,) = breeze.runtimes
    runtime.frames_wanted = 10_000  # never emits its end token
    info = _synth(handle, tmp_path, text="Yes.")
    cap = w.max_frames_for("Yes.", 200)
    assert info["frames"] == cap and info["hit_cap"] is True
    assert sf.read(tmp_path / "o.wav")[0].size == cap * 1920  # every frame up to the cap is kept
    assert runtime.closed == ["seiyuu-1"]  # the codec request was released
    # the worker keeps serving: the next line is normal
    runtime.frames_wanted = 3
    assert _synth(handle, tmp_path)["hit_cap"] is False


def test_the_cap_counts_the_longer_prompt_under_guidance(breeze, tmp_path):
    handle = _handle(tmp_path)
    handle.load()
    (runtime,) = breeze.runtimes
    runtime.frames_wanted = 10_000
    breeze.prompt_len = 1900
    text = "x" * 2000
    assert _synth(handle, tmp_path, text=text)["frames"] == 2048 - 1 - 1900
    # the negative prompt is 7 tokens longer in the fake: both rows are padded to it
    directed = _synth(handle, tmp_path, text=text, instruction="calm")
    assert directed["frames"] == 2048 - 1 - 1907 and directed["hit_cap"] is True
    breeze.prompt_len = 2040
    with pytest.raises(ValueError, match="prompt too long"):
        _synth(handle, tmp_path, text=text)
    assert len(runtime.calls) == 2  # refused before generating


def test_no_frames_writes_an_empty_file(breeze, tmp_path):
    handle = _handle(tmp_path)
    handle.load()
    breeze.runtimes[0].frames_wanted = 0  # the first token was the end token
    info = _synth(handle, tmp_path)
    assert info["frames"] == 0 and info["seconds"] == 0.0 and info["hit_cap"] is False
    assert sf.read(tmp_path / "o.wav")[0].size == 0


def test_a_missing_transcript_is_refused_before_loading(breeze, tmp_path):
    handle = _handle(tmp_path)
    for missing in ("", "   ", None):
        with pytest.raises(ValueError, match="exact transcript"):
            _synth(handle, tmp_path, ref_text=missing)
    assert breeze.runtimes == [] and breeze.events == []
