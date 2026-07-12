"""GPU resource manager: unload-on-handoff, lazy release, and the Ollama unload poll."""

import pytest

from seiyuu.gpu import GpuBusyError, GpuResourceManager


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


# --- Cross-process gpu.lock (two manager instances simulate CLI + server: msvcrt/flock
# locks are per-descriptor, so same-process contention exercises the real path) ---


def test_singleton_is_armed_with_the_data_dir_gpu_lock():
    from seiyuu.gpu import get_gpu_manager

    mgr = get_gpu_manager()
    assert mgr._card_lock is not None
    assert mgr._card_lock.path == get_settings().data_dir / "gpu.lock"


def test_distinct_lock_paths_do_not_interfere(tmp_path):
    m1 = GpuResourceManager(lock_path=tmp_path / "one.lock")
    m2 = GpuResourceManager(lock_path=tmp_path / "two.lock")
    a, b = FakeConsumer(), FakeConsumer()
    with m1.acquire(a, "a"):
        pass
    with m2.acquire(b, "b"):  # different card claims — must not refuse each other
        pass
    assert m1.resident == "a" and m2.resident == "b"


def test_second_process_refused_while_first_is_idle_but_resident(tmp_path):
    lock = tmp_path / "gpu.lock"
    server = GpuResourceManager(lock_path=lock)
    cli = GpuResourceManager(lock_path=lock)
    a, b = FakeConsumer(), FakeConsumer()
    with server.acquire(a, "engine:kokoro"):
        pass
    # the server's job FINISHED (lazy release keeps the model on the card) — the CLI
    # must be refused now, not just mid-job: that idle VRAM is exactly what would OOM
    with pytest.raises(GpuBusyError, match="another seiyuu process holds the GPU"):
        with cli.acquire(b, "engine:chatterbox"):
            pass
    assert cli.resident is None and b.unloads == 0  # refusal left the CLI manager untouched
    assert server.resident == "engine:kokoro"


def test_refusal_names_the_load_and_the_way_out(tmp_path):
    lock = tmp_path / "gpu.lock"
    m1, m2 = GpuResourceManager(lock_path=lock), GpuResourceManager(lock_path=lock)
    with m1.acquire(FakeConsumer(), "llm:ollama:qwen2.5:7b"):
        pass
    with pytest.raises(GpuBusyError) as ei:
        with m2.acquire(FakeConsumer(), "engine:kokoro"):
            pass
    msg = str(ei.value)
    assert "engine:kokoro" in msg  # what was refused
    assert "API server" in msg and "wait" in msg  # the situation and the way out


def test_free_all_releases_the_card_for_the_second_process(tmp_path):
    lock = tmp_path / "gpu.lock"
    server = GpuResourceManager(lock_path=lock)
    cli = GpuResourceManager(lock_path=lock)
    a, b = FakeConsumer(), FakeConsumer()
    with server.acquire(a, "a"):
        pass
    server.free_all()
    assert a.unloads == 1
    with cli.acquire(b, "b"):
        pass
    assert cli.resident == "b"
    cli.free_all()
    with server.acquire(a, "a"):  # and back again — release is a full round trip
        pass


def test_exception_during_load_then_free_all_releases(tmp_path):
    lock = tmp_path / "gpu.lock"
    m1, m2 = GpuResourceManager(lock_path=lock), GpuResourceManager(lock_path=lock)
    a = FakeConsumer()
    with pytest.raises(RuntimeError, match="weights download died"):
        with m1.acquire(a, "engine:kokoro"):
            raise RuntimeError("weights download died")
    # the failed load may have left VRAM allocated, so the claim survives the exception…
    with pytest.raises(GpuBusyError):
        with m2.acquire(FakeConsumer(), "b"):
            pass
    m1.free_all()  # …and the failure path's free_all (warmup handler, render finally) frees it
    with m2.acquire(FakeConsumer(), "b"):
        pass
    assert m2.resident == "b"


def test_contended_acquire_raises_before_touching_state(tmp_path):
    lock = tmp_path / "gpu.lock"
    m1, m2 = GpuResourceManager(lock_path=lock), GpuResourceManager(lock_path=lock)
    with m1.acquire(FakeConsumer(), "a"):
        pass
    with pytest.raises(GpuBusyError):
        with m2.acquire(FakeConsumer(), "b"):
            pass
    m2.free_all()  # refused manager's free_all is a harmless no-op…
    with pytest.raises(GpuBusyError):  # …and must NOT have stolen or broken m1's claim
        with m2.acquire(FakeConsumer(), "b"):
            pass


def test_in_process_handoff_keeps_the_card_claimed(tmp_path):
    lock = tmp_path / "gpu.lock"
    m1, m2 = GpuResourceManager(lock_path=lock), GpuResourceManager(lock_path=lock)
    a, b = FakeConsumer(), FakeConsumer()
    with m1.acquire(a, "a"):
        pass
    with m1.acquire(b, "b"):  # same-process handoff: unload a, keep the OS claim
        pass
    assert a.unloads == 1
    with pytest.raises(GpuBusyError):
        with m2.acquire(FakeConsumer(), "c"):
            pass
    m1.free_all()
    with m2.acquire(FakeConsumer(), "c"):
        pass


def test_identity_reacquire_with_lock_path_stays_a_noop(tmp_path):
    m1 = GpuResourceManager(lock_path=tmp_path / "gpu.lock")
    a = FakeConsumer()
    with m1.acquire(a, "a"):
        pass
    with m1.acquire(a, "a"):  # lazy release + identity re-acquire: unchanged in-process
        pass
    assert a.unloads == 0 and m1.resident == "a"


def test_unload_failure_keeps_the_claim_truthful(tmp_path):
    class ExplodingConsumer:
        def unload(self) -> None:
            raise RuntimeError("CUDA context wedged")

    lock = tmp_path / "gpu.lock"
    m1, m2 = GpuResourceManager(lock_path=lock), GpuResourceManager(lock_path=lock)
    with m1.acquire(ExplodingConsumer(), "engine:bad"):
        pass
    with pytest.raises(RuntimeError, match="CUDA context wedged"):
        m1.free_all()
    # the model is still on the card, so the claim must survive the failed unload
    with pytest.raises(GpuBusyError):
        with m2.acquire(FakeConsumer(), "b"):
            pass
