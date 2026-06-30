"""GPU resource manager: unload-on-handoff, lazy release, and the Ollama unload poll."""

import pytest

from seiyuu.gpu import GpuResourceManager


class FakeConsumer:
    def __init__(self) -> None:
        self.unloads = 0

    def unload(self) -> None:
        self.unloads += 1


def test_unloads_resident_on_handoff_to_a_different_consumer():
    mgr = GpuResourceManager()
    a, b = FakeConsumer(), FakeConsumer()
    with mgr.acquire(a, "a"):
        pass
    assert a.unloads == 0 and mgr.resident == "a"  # lazy: stays resident after its work
    with mgr.acquire(b, "b"):
        assert a.unloads == 1  # the competitor's acquire freed A
    assert mgr.resident == "b" and b.unloads == 0


def test_same_consumer_reacquire_does_not_unload():
    mgr = GpuResourceManager()
    a = FakeConsumer()
    with mgr.acquire(a, "a"):
        pass
    with mgr.acquire(a, "a"):
        pass
    assert a.unloads == 0  # back-to-back use of the same model is free


def test_free_all_unloads_resident():
    mgr = GpuResourceManager()
    a = FakeConsumer()
    with mgr.acquire(a, "a"):
        pass
    mgr.free_all()
    assert a.unloads == 1 and mgr.resident is None
    mgr.free_all()  # idempotent
    assert a.unloads == 1


# --- Ollama provider unload (HTTP mocked) ---

from seiyuu.attribute.providers.local import OllamaProvider  # noqa: E402
from seiyuu.settings import get_settings  # noqa: E402

PROMPTS_DIR = get_settings().prompts_dir


def test_ollama_unload_requests_keep_alive_zero_and_polls_until_gone():
    posts, ps_calls = [], {"n": 0}

    def post(url, payload, timeout):
        posts.append((url, payload))
        return {}

    def get(url, timeout):
        ps_calls["n"] += 1
        # loaded on the first poll, gone on the second
        return {"models": [{"model": "qwen2.5:7b"}]} if ps_calls["n"] == 1 else {"models": []}

    provider = OllamaProvider(
        model="qwen2.5:7b", prompts_dir=PROMPTS_DIR, unload_poll_timeout=5.0, post=post, get=get
    )
    provider.unload()
    assert posts[0][0].endswith("/api/generate")
    assert posts[0][1]["keep_alive"] == 0 and posts[0][1]["model"] == "qwen2.5:7b"
    assert ps_calls["n"] == 2  # polled until the model disappeared


def test_ollama_unload_times_out_loudly_if_never_freed():
    def get(url, timeout):
        return {"models": [{"model": "qwen2.5:7b"}]}  # never frees

    provider = OllamaProvider(
        model="qwen2.5:7b",
        prompts_dir=PROMPTS_DIR,
        unload_poll_timeout=0.0,  # one check then give up
        post=lambda *a, **k: {},
        get=get,
    )
    with pytest.raises(Exception, match="did not unload"):
        provider.unload()


def test_default_unload_is_noop_on_base_engine_and_provider():
    # The ABCs provide a no-op unload so cloud engines / anthropic need not implement it.
    from fake_engine import FakeEngine

    FakeEngine().unload()  # must not raise
