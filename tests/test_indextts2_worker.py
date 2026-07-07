"""IndexTTS-2 worker protocol tests — no torch, no SDK, no GPU.

The worker's request dispatch (handle/serve) and its OOM/stdout-purity guarantees are exercised
with a fake model. The real IndexTTS-2 load/infer path only runs in the separate cu128 env and is
covered by a gated smoke test there, never in the default CPU suite.
"""

import io
import json

from seiyuu.engines import indextts2_worker as w


class _FakeModel:
    def __init__(self, *, fail=None, oom=False):
        self.loaded = False
        self.calls = []
        self.emptied = 0
        self._fail = fail  # exception to raise from synthesize
        self._oom = oom

    def load(self):
        self.loaded = True

    def synthesize(self, *, text, reference_wav, out_path, seed, emo_vector, emo_alpha):
        self.calls.append(
            {
                "text": text,
                "reference_wav": reference_wav,
                "out_path": out_path,
                "seed": seed,
                "emo_vector": emo_vector,
                "emo_alpha": emo_alpha,
            }
        )
        if self._oom:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        if self._fail is not None:
            raise self._fail
        return w.SAMPLE_RATE

    def empty_cache(self):
        self.emptied += 1


def test_ping_and_load():
    model = _FakeModel()
    assert w.handle({"cmd": "ping"}, model) == {"ok": True, "protocol": w.PROTOCOL_VERSION}
    assert w.handle({"cmd": "load"}, model) == {"ok": True}
    assert model.loaded


def test_synthesize_forwards_all_fields_and_returns_sample_rate():
    model = _FakeModel()
    reply = w.handle(
        {
            "cmd": "synthesize",
            "text": "Hello.",
            "reference_wav": "/v/x/reference.wav",
            "out_path": "/tmp/out.wav",
            "seed": 7,
            "emo_vector": [0.0, 0.64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "emo_alpha": 0.8,
        },
        model,
    )
    assert reply == {"ok": True, "sample_rate": 22050, "path": "/tmp/out.wav"}
    call = model.calls[0]
    assert call["seed"] == 7 and call["emo_alpha"] == 0.8
    assert call["emo_vector"][1] == 0.64  # ANGRY dim forwarded verbatim


def test_synthesize_neutral_passes_none_emotion():
    model = _FakeModel()
    w.handle(
        {"cmd": "synthesize", "text": "Hi.", "reference_wav": "r", "out_path": "o"},
        model,
    )
    call = model.calls[0]
    assert call["emo_vector"] is None and call["emo_alpha"] is None and call["seed"] is None


def test_unknown_cmd_is_a_structured_error():
    reply = w.handle({"cmd": "bogus"}, _FakeModel())
    assert reply["ok"] is False and "bogus" in reply["error"]


def test_is_oom_detection():
    assert w._is_oom(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))

    class _OutOfMemoryError(Exception):  # torch.cuda.OutOfMemoryError-shaped: match by type name
        pass

    assert w._is_oom(_OutOfMemoryError("boom"))
    assert not w._is_oom(RuntimeError("some other failure"))


def test_serve_wraps_oom_and_frees_cache():
    model = _FakeModel(oom=True)
    stdin = io.StringIO(
        json.dumps({"cmd": "synthesize", "text": "t", "reference_wav": "r", "out_path": "o"}) + "\n"
    )
    stdout = io.StringIO()
    w.serve(stdin, stdout, model)
    reply = json.loads(stdout.getvalue().strip())
    assert reply["ok"] is False and reply["oom"] is True
    assert "traceback" in reply
    assert model.emptied == 1  # OOM path freed VRAM before replying


def test_serve_reports_non_oom_failure_without_freeing():
    model = _FakeModel(fail=ValueError("bad reference"))
    stdin = io.StringIO(
        json.dumps({"cmd": "synthesize", "text": "t", "reference_wav": "r", "out_path": "o"}) + "\n"
    )
    stdout = io.StringIO()
    w.serve(stdin, stdout, model)
    reply = json.loads(stdout.getvalue().strip())
    assert reply["ok"] is False and reply["oom"] is False
    assert "bad reference" in reply["error"] and model.emptied == 0


def test_serve_skips_blank_lines_and_reports_bad_json():
    model = _FakeModel()
    stdin = io.StringIO("\n   \n" + "not json\n" + json.dumps({"cmd": "ping"}) + "\n")
    stdout = io.StringIO()
    w.serve(stdin, stdout, model)
    lines = [json.loads(x) for x in stdout.getvalue().splitlines() if x.strip()]
    assert lines[0]["ok"] is False and "bad json" in lines[0]["error"]  # only the non-blank garbage
    assert lines[1] == {"ok": True, "protocol": w.PROTOCOL_VERSION}


def test_main_announces_ready_then_exits_on_eof(monkeypatch):
    """main() emits exactly one ready message on the pure-JSON channel, then returns at EOF —
    without loading any weights (the fake stdin has no commands)."""
    monkeypatch.setattr(w.sys, "stdin", io.StringIO(""))
    out = io.StringIO()
    monkeypatch.setattr(w.sys, "stdout", out)
    w.main(["--checkpoints", "/nonexistent"])  # no load/synthesize -> torch never imported
    ready = json.loads(out.getvalue().strip())
    assert ready == {"ok": True, "event": "ready", "protocol": w.PROTOCOL_VERSION}
