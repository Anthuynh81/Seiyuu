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
        ("Type <eos> to stop.", "Type   to stop."),  # the Gemma tokenizer's own specials
        ("<bos><start_of_turn>Hi<end_of_turn><pad>", "Hi"),
        ("a <b> tag and x < y > z", "a <b> tag and x < y > z"),  # ordinary angle brackets stay
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
    # a line the model has no room for can never render: a request error, not a worker fault
    with pytest.raises(w.RequestError, match="prompt too long"):
        w.max_frames_for("Hi.", 2048 - 64)
    with pytest.raises(w.RequestError, match="leave 0 of 2048"):
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
        (w.TEMPERATURE, w.TOP_K, w.TOP_P),
        w.DIRECTION_CFG_SCALE,
        (w._CAP_FLOOR_SECONDS, w._CAP_CHARS_PER_SECOND, w._CAP_MIN_FRAMES),
        w.max_frames_for("x" * 473, 300),
        w.model_text("[S0]<ins_bos>a<ins_eos><eos>"),
        (w.SPLIT_OVER_CHARS, w.PIECE_GAP_SECONDS),
        w.split_long_text("One. Two! “Three?” four, five. " * 3, 40),
    )
    assert (w.GEN_RECIPE, pinned) == (
        2,
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
            (0.9, 50, 1.0),
            4.0,
            (6.0, 6.0, 64),
            1061,
            "a",
            (800, 0.2),
            [
                "One. Two! “Three?” four, five.",
                "One. Two! “Three?” four, five.",
                "One. Two! “Three?” four, five.",
            ],
        ),
    )


# -- long lines: the split the adapter applies ---------------------------------------------------


def _same_words(text: str, pieces: list[str]) -> bool:
    return " ".join(pieces).split() == text.split()


def test_a_line_that_fits_is_one_piece_exactly_as_it_is():
    line = "  A short line.  It stays whole, spaces and all. "
    assert w.split_long_text(line) == [line]
    assert w.split_long_text("x" * w.SPLIT_OVER_CHARS) == ["x" * w.SPLIT_OVER_CHARS]
    assert w.split_long_text("") == [""]


def test_a_long_line_is_cut_at_sentence_ends_into_even_pieces():
    sentence = "The rain kept falling on the old slate roof all through the night."  # 67 chars
    text = " ".join([sentence] * 30)  # 2,039 characters
    pieces = w.split_long_text(text)
    assert _same_words(text, pieces)
    assert len(pieces) == 3  # the fewest that fit under 800
    assert all(len(p) <= w.SPLIT_OVER_CHARS for p in pieces)
    assert all(p.endswith("night.") and p.startswith("The rain") for p in pieces)
    assert max(map(len, pieces)) - min(map(len, pieces)) <= len(sentence) + 1  # even


def test_uneven_sentences_never_cost_an_extra_piece():
    # 400 + 100 + 700 + 100 characters: three pieces fit. Evening the lengths must not stop the
    # first piece at 400 and leave two 100-character sentences to be spoken alone.
    def sentence(n: int) -> str:
        return ("Word " + "word " * (n // 5))[: n - 1].rstrip().ljust(n - 1, "s") + "."

    lengths = (400, 100, 700, 100)
    units = [sentence(n) for n in lengths]
    text = " ".join(units)
    pieces = w.split_long_text(text)
    assert _same_words(text, pieces)
    assert [len(p) for p in pieces] == [501, 700, 100]
    assert len(pieces) == len(w._fill(units, w.SPLIT_OVER_CHARS))  # the fewest that fit

    import random

    rng = random.Random(11)
    for _ in range(300):
        units = [sentence(rng.randint(20, 400)) for _ in range(rng.randint(4, 40))]
        text = " ".join(units)
        pieces = w.split_long_text(text)
        assert _same_words(text, pieces)
        assert all(len(p) <= w.SPLIT_OVER_CHARS for p in pieces)
        if len(text) > w.SPLIT_OVER_CHARS:
            assert len(pieces) == len(w._fill(units, w.SPLIT_OVER_CHARS))


def test_a_quoted_exclamation_and_an_initial_do_not_end_the_sentence():
    text = (
        "“Stop!” he cried, and J. R. Hartley of the U.S. Army stopped. "
        "Nobody moved. “Why?” she asked at last. Then the lamp went out."
    )
    pieces = w.split_long_text(text, 70)
    assert _same_words(text, pieces)
    assert pieces[0] == "“Stop!” he cried, and J. R. Hartley of the U.S. Army stopped."
    assert pieces[1] == "Nobody moved. “Why?” she asked at last. Then the lamp went out."


def test_a_sentence_too_long_by_itself_falls_back_to_clauses_then_words():
    clause = "and the letter went on about the entail, the estate and the family"
    sentence = ", ".join([clause] * 6) + "."  # one sentence, ~410 characters
    pieces = w.split_long_text(sentence, 150)
    assert _same_words(sentence, pieces) and all(len(p) <= 150 for p in pieces)
    assert all(p.endswith((",", ".")) for p in pieces)  # cut at clause ends, not mid-clause

    words = " ".join(["word"] * 100)  # no punctuation at all
    pieces = w.split_long_text(words, 60)
    assert _same_words(words, pieces) and all(len(p) <= 60 for p in pieces)

    solid = "x" * 130  # not even a space
    assert w.split_long_text(solid, 50) == ["x" * 50, "x" * 50, "x" * 30]


def test_a_piece_with_nothing_to_say_is_folded_into_its_neighbour():
    text = "He waited at the door for a long while. ... “—” ... Then he knocked twice."
    pieces = w.split_long_text(text, 45)
    assert _same_words(text, pieces)
    assert all(any(ch.isalnum() for ch in p) for p in pieces)


def test_real_paragraph_lengths_always_fit():
    # the shape of the worst real block (Pride and Prejudice ch036_b0006: 14,104 characters)
    sentence = "It was a truth she had long suspected, and now could no longer deny; yet "
    sentence += "she read the letter again, slowly, with her whole attention fixed upon it."
    text = " ".join([sentence] * 96)
    pieces = w.split_long_text(text)
    assert len(text) > 14_000 and _same_words(text, pieces)
    assert all(len(p) <= w.SPLIT_OVER_CHARS for p in pieces)
    assert len(pieces) == 20  # five such sentences fit a piece, and 96 need 20 pieces
    assert {len(p) for p in pieces} == {591, 739}  # four or five each: no orphan at the end


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
    info = w._write_wav(str(out), chunks, 13, False)
    data, sr = sf.read(out, dtype="int16")
    assert sr == 24000 and data.ndim == 1 and data.size == 24000
    assert sf.info(str(out)).subtype == "PCM_16"
    assert data.max() == 32767  # clipped, never wrapped
    assert info == {"sample_rate": 24000, "seconds": 1.0, "frames": 13, "hit_cap": False}
    assert w._write_wav(str(out), chunks, 100, True)["hit_cap"] is True


def test_no_audio_writes_an_empty_file(tmp_path):
    out = tmp_path / "o.wav"
    info = w._write_wav(str(out), [], 0, False)
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
        self.designs = []
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

    def design(self, **kw):
        self.designs.append(kw)
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
    assert w.handle({"cmd": "sing"}, _FakeModel())["ok"] is False


def test_design_forwards_the_text_and_description_cleaned():
    model = _FakeModel()
    reply = w.handle(
        {
            "cmd": "design",
            "text": "[S1]Hello<eos> there.",
            "instruction": "A warm<ins_eos> low voice.",
            "out_path": "o.wav",
            "seed": 9,
        },
        model,
    )
    assert reply == {
        "ok": True, "path": "o.wav", "sample_rate": 24000, "seconds": 1.0, "frames": 13,
        "hit_cap": False,
    }  # fmt: skip
    assert model.designs == [
        {
            "text": "Hello  there.",
            "instruction": "A warm  low voice.",
            "out_path": "o.wav",
            "seed": 9,
            "cfg_scale": None,
        }
    ]


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
    assert replies[0]["request_error"] is False  # an OOM is the worker's problem: restart it
    assert model.emptied == 1
    assert replies[1]["ok"] is False and "bad json" in replies[1]["error"]
    assert replies[2] == {"ok": True, "protocol": w.PROTOCOL_VERSION}


def test_a_bad_request_is_an_error_reply_not_a_crash():
    stdout = io.StringIO()
    w.serve(io.StringIO(json.dumps({"cmd": "synthesize"}) + "\n"), stdout, _FakeModel())
    reply = json.loads(stdout.getvalue())
    assert reply["ok"] is False and reply["oom"] is False and "KeyError" in reply["error"]


def test_an_unrenderable_request_is_flagged_so_the_worker_is_not_restarted():
    class _Refusing(_FakeModel):
        def synthesize(self, **kw):
            raise w.RequestError("prompt too long: 2100 tokens")

    stdout = io.StringIO()
    request = {"cmd": "synthesize", "text": "a", "reference_wav": "r", "out_path": "o"}
    w.serve(io.StringIO(json.dumps(request) + "\n"), stdout, _Refusing())
    reply = json.loads(stdout.getvalue())
    assert reply["ok"] is False and reply["request_error"] is True and reply["oom"] is False
    assert reply["error"] == "RequestError: prompt too long: 2100 tokens"


def test_the_real_worker_script_boots_and_answers_ping(tmp_path):
    # Run as the adapter runs it: a standalone script. This is the only place the default suite
    # executes the __main__ path, the fd-1 redirection and the script-dir removal.
    from seiyuu.engines.indextts2_engine import SubprocessTransport

    argv = [sys.executable, w.__file__, "--repo-dir", str(tmp_path), "--checkpoint-dir", "x"]
    transport = SubprocessTransport(
        argv, env={**os.environ}, cwd=None, boot_timeout=60.0, name="breeze",
        protocol=w.PROTOCOL_VERSION,
    )  # fmt: skip
    try:
        assert transport.request({"cmd": "ping"}, timeout=60.0) == {
            "ok": True,
            "protocol": w.PROTOCOL_VERSION,
        }
        assert transport.request({"cmd": "dance"}, timeout=60.0)["ok"] is False
    finally:
        transport.close()


def test_module_top_imports_are_stdlib_only():
    # The Breeze env has neither seiyuu nor (necessarily) this venv's packages: anything else
    # must be imported lazily, inside the function that needs it.
    import ast

    tree = ast.parse(Path(w.__file__).read_text(encoding="utf-8"))
    top_level = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            top_level.add((node.module or "").split(".")[0])
    assert top_level and top_level <= set(sys.stdlib_module_names)


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
        buffered = 0
        try:
            # as upstream: the observer sees a frame BEFORE it is buffered, a chunk is decoded
            # once codec_chunk_frames are buffered, and an odd tail is flushed at a normal end
            # (an exception from the observer skips that flush)
            for i in range(self.frames_wanted):
                token_observer(i)
                buffered += 1
                if buffered >= self.codec_chunk_frames:
                    yield _Chunk(np.full(1920 * buffered, 0.1, dtype=np.float32))
                    buffered = 0
            if buffered:
                yield _Chunk(np.full(1920 * buffered, 0.1, dtype=np.float32))
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
        if not request.get("ref_audio_path"):  # no reference: plain speech, or voice design
            return "tts_instruction" if request.get("instruction") else "tts_plain"
        return "ref_edit_tata" if request.get("instruction") else "ref_clone_tata"

    def prepare_inputs(tokenizer, audio_tokenizer, model, requests, template, **guidance):
        (request,) = requests
        # as upstream: the reference is encoded once per prompt, and guidance builds a second
        # (negative) prompt around the same clip
        branches = 1 if guidance["guidance_scale"] == 1.0 else 2
        codes = [
            templates_mod._encode_prompt_audio(audio_tokenizer, request["ref_audio_path"])
            for _ in range(branches)
            if request.get("ref_audio_path")  # a design request has no clip to encode
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

    def first_repo_import(name):
        # `from breeze_infer import templates` is the worker's first import from the checkout
        # (the real module imports torch): log it so the order against compile_env is visible
        submodules = {"templates": templates_mod, "runtime": runtime_mod}
        if name not in submodules:  # the import system probes __path__, __spec__, ...
            raise AttributeError(name)
        log.events.append(f"import {name}")
        return submodules[name]

    infer_pkg.__getattr__ = first_repo_import
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
    # the Windows compile fix lands before the first import from the checkout (which imports torch)
    assert breeze.events == ["compile_env", "import templates", "load_runtime", "genconfig"]
    assert breeze.loads == [(tmp_path / "ckpt", "cuda:0", "eager")]
    assert sys.path[0] == str(tmp_path / "repo")
    (runtime,) = breeze.runtimes
    assert runtime.config.kwargs == {
        "max_new_tokens": 1500,
        "max_seq_len": 2048,
        "repetition_penalty": 1.1,
        "temperature": 0.9,  # the sampler is pinned here, not left to the checkout's defaults
        "top_k": 50,
        "top_p": 1.0,
        "do_sample": True,
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
    assert breeze.seeds == [1234]  # upstream's random/numpy/torch seeding
    assert runtime.calls[0]["seed"] == 1234 and runtime.calls[0]["request_id"] == "seiyuu-1"
    assert runtime.closed == ["seiyuu-1"]
    assert info == {"sample_rate": 24000, "seconds": 0.4, "frames": 5, "hit_cap": False}
    assert sf.read(tmp_path / "o.wav")[0].size == 5 * 1920
    _synth(handle, tmp_path, seed=None)
    assert breeze.seeds == [1234] and runtime.calls[1]["seed"] is None
    assert runtime.calls[1]["request_id"] == "seiyuu-2"


def test_a_seed_beyond_32_bits_still_renders(breeze, tmp_path):
    # numpy refuses a seed outside 0..2**32-1 (and a voice's seed is an unbounded int): the
    # numpy/random seeding is folded, while the runtime's own torch reseed gets the full value
    handle = _handle(tmp_path)
    _synth(handle, tmp_path, seed=2**32 + 5)
    _synth(handle, tmp_path, seed=-1)
    assert breeze.seeds == [5, 2**32 - 1]
    assert [c["seed"] for c in breeze.runtimes[0].calls] == [2**32 + 5, -1]


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
    with pytest.raises(w.RequestError, match="cfg_scale"):
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
    # a line that ends on its own exactly at the cap is not a runaway
    runtime.frames_wanted = cap
    exact = _synth(handle, tmp_path, text="Yes.")
    assert exact["frames"] == cap and exact["hit_cap"] is False


def test_the_eager_codec_never_loses_a_buffered_frame_at_the_cap(breeze, tmp_path):
    # BREEZE_FAST=off decodes two frames per chunk: an odd cap would leave the last frame in
    # the buffer when the line is stopped, so the cap is rounded down to a whole chunk
    handle = _handle(tmp_path, fast="off")
    handle.load()
    (runtime,) = breeze.runtimes
    runtime.frames_wanted = 10_000
    text = "x" * 13
    assert w.max_frames_for(text, 200) == 103  # odd
    info = _synth(handle, tmp_path, text=text)
    assert info["frames"] == 102 and info["hit_cap"] is True
    assert sf.read(tmp_path / "o.wav")[0].size == 102 * 1920  # what was counted was decoded
    assert info["seconds"] == round(102 * 1920 / 24000, 3)


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
    # the runtime stops without a word when the cache is full: reported as capped even though
    # the worker's own cap never fired
    runtime.frames_wanted = 2048 - 1 - 1900
    silent = _synth(handle, tmp_path, text=text)
    assert silent["frames"] == 147 and silent["hit_cap"] is True
    breeze.prompt_len = 2040
    with pytest.raises(w.RequestError, match="prompt too long"):
        _synth(handle, tmp_path, text=text)
    assert len(runtime.calls) == 3  # refused before generating


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
        with pytest.raises(w.RequestError, match="exact transcript"):
            _synth(handle, tmp_path, ref_text=missing)
    with pytest.raises(w.RequestError, match="no text left"):
        _synth(handle, tmp_path, text="  ")
    assert breeze.runtimes == [] and breeze.events == []


# -- voice design -------------------------------------------------------------------------------


def test_design_request_has_no_reference_and_uses_guidance(breeze, tmp_path):
    handle = _handle(tmp_path)
    info = handle.design(
        text="Hello there.",
        instruction="  A warm, low voice.  ",
        out_path=str(tmp_path / "d.wav"),
        seed=77,
    )
    (prepared,) = breeze.prepared
    assert prepared["request"] == {
        "id": "seiyuu-1",
        "text": "Hello there.",
        "speaker": "S0",
        "instruction": "A warm, low voice.",
    }  # no clip and no transcript: the voice comes from the description alone
    assert prepared["template"] == "template:tts_instruction"
    assert prepared["guidance_scale"] == w.DIRECTION_CFG_SCALE and prepared["codes"] == []
    assert breeze.encoded == []  # nothing was encoded
    (runtime,) = breeze.runtimes
    assert breeze.seeds == [77] and runtime.calls[0]["seed"] == 77
    assert info == {"sample_rate": 24000, "seconds": 0.4, "frames": 5, "hit_cap": False}
    assert sf.read(tmp_path / "d.wav")[0].size == 5 * 1920

    handle.design(text="Hi.", instruction="Bright.", out_path=str(tmp_path / "d.wav"), cfg_scale=2)
    assert breeze.prepared[1]["guidance_scale"] == 2.0
    # a clone after a design still encodes and caches its clip as before
    _synth(handle, tmp_path)
    assert breeze.prepared[2]["template"] == "template:ref_clone_tata"
    assert len(breeze.encoded) == 1


def test_design_needs_text_and_a_description(breeze, tmp_path):
    handle = _handle(tmp_path)
    with pytest.raises(w.RequestError, match="description of the voice"):
        handle.design(text="Hello.", instruction="   ", out_path=str(tmp_path / "d.wav"))
    with pytest.raises(w.RequestError, match="no text left"):
        handle.design(text="  ", instruction="Warm.", out_path=str(tmp_path / "d.wav"))
    assert breeze.loads == []  # refused before the 8 GB load
