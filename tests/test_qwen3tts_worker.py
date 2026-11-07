"""Qwen3-TTS worker tests — no SDK, no GPU.

The request dispatch (handle/serve) runs against a fake model, and ``ModelHandle`` runs against a
fake faster-qwen3-tts object, so the clone/design call shapes (ICL vs x-vector, forced
non-streaming, the per-call seed, the prompt cache, one checkpoint resident) are pinned without
the real package. The real load/generate path only runs in the separate env.
"""

import io
import json
import sys
import types

import numpy as np
import pytest
import soundfile as sf
import torch

from seiyuu.engines import qwen3tts_worker as w

# -- pure helpers -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('"Hello there."', "Hello there."),
        ("“Wait,” ", "Wait,"),
        ("  “Really?”", "Really?"),
        ("He said nothing.", "He said nothing."),
        ("'Cause I said so.", "'Cause I said so."),  # a leading apostrophe is an elision
        ('"', '"'),  # nothing left: keep the original (base.synthesize handles silence)
    ],
)
def test_model_text_strips_a_wrapping_quote_pair(text, expected):
    assert w.model_text(text) == expected


def test_max_new_tokens_scales_with_text():
    assert w.max_new_tokens_for("Yes.", 2048) == 64  # the floor (4.5 s would be 57 frames)
    assert w.max_new_tokens_for("x" * 200, 2048) == int(np.ceil((4 + 200 / 8) * 12.5))
    # over 2x the measured pace: a 473-char line spoke in 23.3 s (291 frames) on the 3080 Ti
    assert w.max_new_tokens_for("x" * 473, 2048, prompt_frames=174, prompt_chars=226) > 2 * 291


def test_max_new_tokens_leaves_room_for_the_prompt_in_the_static_cache():
    text = "x" * 3000
    alone = w.max_new_tokens_for(text, 2048)
    assert alone == 2048 - 1 - (16 + 1000)  # the target text itself is prefilled
    with_ref = w.max_new_tokens_for(text, 2048, prompt_frames=200, prompt_chars=300)
    assert with_ref == alone - 200 - 100
    assert w.max_new_tokens_for(text, 4096) == 4096 - 1 - 1016
    # never below the floor, even when the prompt alone overflows the cache
    assert w.max_new_tokens_for("x" * 9000, 2048) == 64


def test_write_wav_is_int16_mono_24k_and_flags_the_cap(tmp_path):
    out = tmp_path / "o.wav"
    info = w._write_wav(str(out), np.full(24000, 0.5, dtype=np.float32), 24000, 100)
    data, sr = sf.read(out, dtype="int16")
    assert sr == 24000 and data.ndim == 1 and data.size == 24000
    assert sf.info(str(out)).subtype == "PCM_16"
    assert info == {"sample_rate": 24000, "seconds": 1.0, "hit_cap": False}
    capped = w._write_wav(str(out), np.zeros(int(24000 * 8), dtype=np.float32), 24000, 100)
    assert capped["hit_cap"] is True


def test_no_tokens_writes_an_empty_file(tmp_path):
    out = tmp_path / "o.wav"
    w._write_wav(str(out), np.zeros(1, dtype=np.float32), 24000, 100)
    assert sf.read(out)[0].size == 0  # the adapter fails an empty file loudly


# -- dispatch ---------------------------------------------------------------------------------


class _FakeModel:
    def __init__(self, *, oom=False):
        self.calls = []
        self.loaded = []
        self.emptied = 0
        self._oom = oom

    def load(self, kind="base"):
        self.loaded.append(kind)

    def synthesize(self, **kw):
        self.calls.append(("synthesize", kw))
        if self._oom:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return {"sample_rate": 24000, "seconds": 1.0, "hit_cap": False}

    def design(self, **kw):
        self.calls.append(("design", kw))
        return {"sample_rate": 24000, "seconds": 2.0, "hit_cap": False}

    def empty_cache(self):
        self.emptied += 1


def test_ping_and_load():
    model = _FakeModel()
    assert w.handle({"cmd": "ping"}, model) == {"ok": True, "protocol": w.PROTOCOL_VERSION}
    assert w.handle({"cmd": "load"}, model) == {"ok": True, "model": "base"}
    assert w.handle({"cmd": "load", "model": "design"}, model)["model"] == "design"
    assert model.loaded == ["base", "design"]


def test_synthesize_strips_quotes_and_forwards_everything():
    model = _FakeModel()
    reply = w.handle(
        {
            "cmd": "synthesize",
            "text": '"Hi."',
            "reference_wav": "/v/x/reference.wav",
            "reference_sha256": "abc",
            "ref_text": "I keep a notebook.",
            "out_path": "/tmp/o.wav",
            "seed": 7,
            "gen": {"temperature": 0.8},
        },
        model,
    )
    assert reply == {
        "ok": True,
        "path": "/tmp/o.wav",
        "sample_rate": 24000,
        "seconds": 1.0,
        "hit_cap": False,
    }
    _, kw = model.calls[0]
    assert kw["text"] == "Hi."
    assert kw["reference_sha256"] == "abc" and kw["ref_text"] == "I keep a notebook."
    assert kw["seed"] == 7 and kw["gen"] == {"temperature": 0.8}


def test_design_dispatch():
    model = _FakeModel()
    reply = w.handle(
        {"cmd": "design", "text": "Hello.", "instruct": "old man", "out_path": "o", "seed": 3},
        model,
    )
    assert reply["ok"] and reply["seconds"] == 2.0
    assert model.calls == [
        (
            "design",
            {"text": "Hello.", "instruct": "old man", "out_path": "o", "seed": 3, "gen": None},
        )
    ]


def test_unknown_cmd():
    assert w.handle({"cmd": "dance"}, _FakeModel())["ok"] is False


def test_serve_reports_oom_and_keeps_going():
    model = _FakeModel(oom=True)
    stdin = io.StringIO(
        json.dumps({"cmd": "synthesize", "text": "a", "reference_wav": "r", "out_path": "o"})
        + "\nnot json\n"
        + json.dumps({"cmd": "ping"})
        + "\n"
    )
    stdout = io.StringIO()
    w.serve(stdin, stdout, model)
    replies = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert replies[0]["ok"] is False and replies[0]["oom"] is True
    assert model.emptied == 1
    assert replies[1]["ok"] is False and "bad json" in replies[1]["error"]
    assert replies[2] == {"ok": True, "protocol": w.PROTOCOL_VERSION}


def test_main_announces_ready_and_exits_at_eof():
    out = io.StringIO()
    w.main(["--models-dir", "/m", "--backend", "stock"], stdin=io.StringIO(""), stdout=out)
    assert json.loads(out.getvalue()) == {
        "ok": True,
        "event": "ready",
        "protocol": w.PROTOCOL_VERSION,
    }


# -- ModelHandle against a fake faster-qwen3-tts ----------------------------------------------


class _FakeQwen:
    """Stands in for qwen_tts.Qwen3TTSModel (the fast wrapper's ``.model``)."""

    def __init__(self):
        self.prompts = []

    def create_voice_clone_prompt(self, *, ref_audio, ref_text, x_vector_only_mode):
        self.prompts.append(
            {"ref_audio": ref_audio, "ref_text": ref_text, "xvec": x_vector_only_mode}
        )
        return [f"prompt{len(self.prompts)}"]


class _FakeFast:
    def __init__(self, path=None, **kwargs):
        self.path, self.kwargs = path, kwargs
        self.model = _FakeQwen()
        self.calls = []
        self.warmed = 0

    def warmup(self):
        self.warmed += 1

    @classmethod
    def from_pretrained(cls, *, model_name, **kwargs):
        built = cls(model_name, **kwargs)
        _FakeFast.built.append(built)
        return built

    def _record(self, kind, kwargs):
        self.calls.append({"kind": kind, "seed": torch.initial_seed(), **kwargs})
        return [np.full(2400, 0.1, dtype=np.float32)], 24000

    def generate_voice_clone(self, **kwargs):
        return self._record("clone", kwargs)

    def generate_voice_design(self, **kwargs):
        return self._record("design", kwargs)


_FakeFast.built = []


@pytest.fixture
def fake_fast(monkeypatch):
    _FakeFast.built = []
    mod = types.ModuleType("faster_qwen3_tts")
    mod.FasterQwen3TTS = _FakeFast
    monkeypatch.setitem(sys.modules, "faster_qwen3_tts", mod)
    return _FakeFast


def _reference(tmp_path, seconds=1.0, sr=24000):
    path = tmp_path / "reference.wav"
    sf.write(path, np.full(int(seconds * sr), 0.25, dtype=np.float32), sr, subtype="PCM_16")
    return str(path)


def test_load_builds_the_fast_model_from_the_local_dir(fake_fast, tmp_path):
    handle = w.ModelHandle(str(tmp_path), max_seq_len=3072)
    handle.load("base")
    built = fake_fast.built[0]
    assert built.path.endswith(w.CHECKPOINTS["base"])
    assert built.kwargs == {
        "device": "cuda",
        "dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
        "max_seq_len": 3072,
    }
    assert built.warmed == 1  # CUDA graphs captured at load, not on the first line
    handle.load("base")
    assert len(fake_fast.built) == 1  # already resident: no reload


def test_one_checkpoint_resident_at_a_time(fake_fast, tmp_path):
    handle = w.ModelHandle(str(tmp_path))
    ref = _reference(tmp_path)
    handle.synthesize(text="Hi.", reference_wav=ref, out_path=str(tmp_path / "a.wav"), seed=1)
    assert handle._prompts
    handle.design(text="Hello.", instruct="old man", out_path=str(tmp_path / "b.wav"), seed=2)
    assert handle.kind == "design" and handle._tts is fake_fast.built[1]
    assert not handle._prompts  # prompts held the old model's tensors
    assert fake_fast.built[1].path.endswith(w.CHECKPOINTS["design"])
    with pytest.raises(ValueError, match="unknown checkpoint"):
        handle.load("custom")


def test_clone_forces_non_streaming_seeds_and_caps(fake_fast, tmp_path):
    handle = w.ModelHandle(str(tmp_path))
    ref = _reference(tmp_path)
    out = tmp_path / "o.wav"
    info = handle.synthesize(
        text="Hello there.",
        reference_wav=ref,
        out_path=str(out),
        reference_sha256="sha1",
        seed=1234,
        gen={"temperature": 0.7, "top_k": 30.0, "do_sample": False, "text": "hijack"},
    )
    call = handle._tts.calls[0]
    assert call["non_streaming_mode"] is True
    assert call["language"] == "English"
    assert call["seed"] == 1234  # seeded right before generating
    assert call["max_new_tokens"] == w.max_new_tokens_for("Hello there.", 2048)
    assert call["temperature"] == 0.7 and call["top_k"] == 30 and isinstance(call["top_k"], int)
    assert "do_sample" not in call and call["text"] == "Hello there."
    assert info["sample_rate"] == 24000 and sf.read(out)[0].size == 2400


def test_x_vector_clone_without_a_transcript(fake_fast, tmp_path):
    handle = w.ModelHandle(str(tmp_path))
    ref = _reference(tmp_path)
    handle.synthesize(text="Hi.", reference_wav=ref, out_path=str(tmp_path / "o.wav"), seed=1)
    prompt = handle._tts.model.prompts[0]
    assert prompt == {"ref_audio": ref, "ref_text": None, "xvec": True}


def test_icl_clone_pads_the_reference_and_caches_by_content(fake_fast, tmp_path):
    handle = w.ModelHandle(str(tmp_path))
    ref = _reference(tmp_path, seconds=1.0)
    out = str(tmp_path / "o.wav")
    for _ in range(2):
        handle.synthesize(
            text="Hi.", reference_wav=ref, out_path=out, reference_sha256="v1", ref_text=" Tx. "
        )
    qwen = handle._tts.model
    assert len(qwen.prompts) == 1  # the second line reuses the encoded reference
    audio, sr = qwen.prompts[0]["ref_audio"]
    assert sr == 24000 and audio.size == int(1.5 * 24000)  # + 0.5 s of silence
    assert np.all(audio[-12000:] == 0)
    assert qwen.prompts[0]["ref_text"] == "Tx." and qwen.prompts[0]["xvec"] is False
    assert handle._tts.calls[1]["voice_clone_prompt"] == ["prompt1"]
    # same path, new audio (a re-clone): a new content hash re-encodes
    handle.synthesize(
        text="Hi.", reference_wav=ref, out_path=out, reference_sha256="v2", ref_text="Tx."
    )
    assert len(qwen.prompts) == 2


def test_the_cap_counts_the_prompt_that_shares_the_cache(fake_fast, tmp_path):
    handle = w.ModelHandle(str(tmp_path), max_seq_len=400)
    ref = _reference(tmp_path, seconds=1.0)  # + 0.5 s pad = 19 codec frames in the ICL prompt
    text = "x" * 300
    out = str(tmp_path / "o.wav")
    handle.synthesize(text=text, reference_wav=ref, out_path=out, ref_text="Tx.")
    handle.synthesize(text=text, reference_wav=ref, out_path=out)  # x-vector: no ref in prompt
    icl, xvec = (c["max_new_tokens"] for c in handle._tts.calls)
    assert icl == 400 - 1 - (16 + 19 + 101) == 263
    assert xvec == 400 - 1 - (16 + 100)
    handle.design(text=text, instruct="y" * 60, out_path=out)
    assert handle._tts.calls[-1]["max_new_tokens"] == 400 - 1 - (16 + 120)


def test_design_call_shape(fake_fast, tmp_path):
    handle = w.ModelHandle(str(tmp_path), language="Auto")
    handle.design(text="Hello.", instruct="old man", out_path=str(tmp_path / "o.wav"), seed=99)
    call = handle._tts.calls[0]
    assert call["kind"] == "design" and call["instruct"] == "old man"
    assert call["language"] == "Auto" and call["non_streaming_mode"] is True
    assert call["seed"] == 99


def test_stock_backend_uses_qwen_tts(monkeypatch, tmp_path):
    built = {}

    class _Stock:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            built.update(path=path, **kwargs)
            return cls()

    mod = types.ModuleType("qwen_tts")
    mod.Qwen3TTSModel = _Stock
    monkeypatch.setitem(sys.modules, "qwen_tts", mod)
    handle = w.ModelHandle(str(tmp_path), backend="stock")
    handle.load("base")
    assert built["device_map"] == "cuda:0" and built["dtype"] is torch.bfloat16
    assert handle._qwen is handle._tts  # no wrapper: the stock model builds prompts itself
