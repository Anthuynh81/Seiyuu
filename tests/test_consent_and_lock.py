"""M6a-5: hash-bound consent attestation, the cross-process file lock, and the
single-voice render's GPU-manager fold."""

import json
import threading
import time

import numpy as np
import pytest

from factories import make_book
from fake_engine import FakeEngine
from seiyuu.engines import AudioFile
from seiyuu.gpu import GpuResourceManager
from seiyuu.render import RenderError, render_book, render_book_multivoice
from seiyuu.repository import FileLockHandle, RepositoryError, file_lock
from seiyuu.voices import (
    ConsentAttestation,
    VoiceAssignment,
    VoiceKind,
    VoiceLibrary,
    VoiceLibraryError,
    VoiceMeta,
    ensure_cloud_voice,
    sha256_file,
)
from test_cloud_voice import FakeClient
from test_render_multivoice import _report

# --- consent attestation ---


def _cloned_meta(vid="elena_x", consent=None, attested=True) -> VoiceMeta:
    return VoiceMeta(
        voice_id=vid, name="Elena", kind=VoiceKind.CLONED, engine="chatterbox",
        reference_audio="reference.wav", consent_attested=attested, consent=consent,
    )  # fmt: skip


def _reference(lib: VoiceLibrary, vid: str, fill=0.1) -> str:
    path = lib.reference_path(vid)
    AudioFile(samples=np.full(2400, fill, dtype=np.float32)).save(path)
    return sha256_file(path)


def test_verify_consent_passes_with_matching_hash(tmp_path):
    lib = VoiceLibrary(tmp_path / "voices")
    digest = _reference(lib, "elena_x")
    meta = _cloned_meta(consent=ConsentAttestation(attested_by="ann", reference_sha256=digest))
    lib.save(meta)
    lib.verify_consent(meta)  # no raise


def test_verify_consent_refuses_swapped_reference_audio(tmp_path):
    """Consent binds to THE audio: replacing reference.wav under an attested voice_id
    must refuse — that is the whole point of the structured record."""
    lib = VoiceLibrary(tmp_path / "voices")
    digest = _reference(lib, "elena_x")
    meta = _cloned_meta(consent=ConsentAttestation(attested_by="ann", reference_sha256=digest))
    lib.save(meta)
    _reference(lib, "elena_x", fill=0.5)  # someone swaps the audio
    with pytest.raises(VoiceLibraryError, match="does not match its consent"):
        lib.verify_consent(meta)


def test_verify_consent_refuses_missing_reference_when_attested(tmp_path):
    lib = VoiceLibrary(tmp_path / "voices")
    meta = _cloned_meta(consent=ConsentAttestation(attested_by="ann", reference_sha256="0" * 64))
    with pytest.raises(VoiceLibraryError, match="missing"):
        lib.verify_consent(meta)


def test_verify_consent_grandfathers_bool_only_metas(tmp_path):
    """Pre-M6a voices carry only the bool — they must keep working without a record."""
    lib = VoiceLibrary(tmp_path / "voices")
    lib.verify_consent(_cloned_meta(consent=None, attested=True))  # no raise
    with pytest.raises(VoiceLibraryError, match="no consent attestation"):
        lib.verify_consent(_cloned_meta(consent=None, attested=False))


def test_verify_consent_ignores_non_cloned(tmp_path):
    lib = VoiceLibrary(tmp_path / "voices")
    preset = VoiceMeta(
        voice_id="p", name="P", kind=VoiceKind.PRESET, engine="kokoro", preset_id="af_heart"
    )
    lib.verify_consent(preset)  # presets never need consent


def test_cloud_upload_refuses_hash_mismatch(tmp_path):
    """ensure_cloud_voice ships reference.wav to a third party — the moment the binding
    must hold. A swapped file refuses before any slot/create traffic."""
    lib = VoiceLibrary(tmp_path / "voices")
    digest = _reference(lib, "elena_x")
    meta = VoiceMeta(
        voice_id="elena_x", name="Elena", kind=VoiceKind.CLONED, engine="elevenlabs",
        reference_audio="reference.wav", consent_attested=True,
        consent=ConsentAttestation(attested_by="ann", reference_sha256=digest),
    )  # fmt: skip
    lib.save(meta)
    _reference(lib, "elena_x", fill=0.9)
    client = FakeClient()
    from seiyuu.voices import CloudVoiceError

    with pytest.raises(CloudVoiceError, match="does not match its consent"):
        ensure_cloud_voice(meta, client, lib, max_slots=10)
    assert client.voices.created == []  # nothing was uploaded


def test_multivoice_render_verifies_consent_via_library(tmp_path, monkeypatch):
    """The render path uses the hash-bound check, not just the flippable bool."""
    monkeypatch.setattr("seiyuu.render.pipeline.get_engine", lambda engine_id, **kw: FakeEngine())
    lib = VoiceLibrary(tmp_path / "voices")
    digest = _reference(lib, "elena_x")
    lib.save(_cloned_meta(consent=ConsentAttestation(attested_by="ann", reference_sha256=digest)))
    lib.save(
        VoiceMeta(voice_id="narrator_v", name="N", kind=VoiceKind.PRESET,
                  engine="kokoro", preset_id="af_heart")
    )  # fmt: skip
    _reference(lib, "elena_x", fill=0.7)  # swap after attestation
    assignment = VoiceAssignment(
        book_id="test-book-00000000",
        narrator_voice_id="narrator_v",
        assignments={"alice": "elena_x"},
    )
    with pytest.raises(RenderError, match="consent"):
        render_book_multivoice(
            _report(), make_book(), lib, assignment, tmp_path / "out", gpu=GpuResourceManager()
        )


def test_clone_cli_writes_attestation_record(tmp_path):
    from click.testing import CliRunner

    from seiyuu.cli import main

    ref = tmp_path / "clip.wav"
    AudioFile(samples=np.zeros(2400, dtype=np.float32)).save(ref)
    result = CliRunner().invoke(
        main,
        [
            "voice",
            "clone",
            "Elena",
            str(ref),
            "--consent",
            "--consent-by",
            "Ann Author",
            "--voice-id",
            "elena_x",
            "--voices-dir",
            str(tmp_path / "voices"),
        ],  # fmt: skip
    )
    assert result.exit_code == 0, result.output
    assert "consent recorded: Ann Author" in result.output
    meta = json.loads((tmp_path / "voices" / "elena_x" / "meta.json").read_text(encoding="utf-8"))
    assert meta["consent"]["attested_by"] == "Ann Author"
    assert meta["consent"]["reference_sha256"] == sha256_file(ref)
    assert meta["consent"]["statement"]
    # and the copied reference verifies end to end
    lib = VoiceLibrary(tmp_path / "voices")
    lib.verify_consent(lib.load("elena_x"))


# --- file lock ---


def test_file_lock_excludes_concurrent_holders(tmp_path):
    lock_path = tmp_path / "x.lock"
    order: list[str] = []
    inside = threading.Event()
    release = threading.Event()

    def holder():
        with file_lock(lock_path):
            order.append("a-in")
            inside.set()
            release.wait(5.0)
            order.append("a-out")

    t = threading.Thread(target=holder, daemon=True)
    t.start()
    assert inside.wait(5.0)
    waiter_done = threading.Event()

    def waiter():
        with file_lock(lock_path):
            order.append("b-in")
        waiter_done.set()

    t2 = threading.Thread(target=waiter, daemon=True)
    t2.start()
    time.sleep(0.15)  # give the waiter a chance to (wrongly) enter
    assert order == ["a-in"]  # excluded while held
    release.set()
    assert waiter_done.wait(5.0)
    assert order == ["a-in", "a-out", "b-in"]
    t.join(5.0), t2.join(5.0)


def test_file_lock_times_out_loudly(tmp_path):
    lock_path = tmp_path / "x.lock"
    inside = threading.Event()
    release = threading.Event()

    def holder():
        with file_lock(lock_path):
            inside.set()
            release.wait(5.0)

    t = threading.Thread(target=holder, daemon=True)
    t.start()
    assert inside.wait(5.0)
    with pytest.raises(RepositoryError, match="could not acquire"):
        with file_lock(lock_path, timeout=0.2):
            pass
    release.set()
    t.join(5.0)


def test_file_lock_and_handle_contend_on_the_same_lock(tmp_path):
    # file_lock is built on FileLockHandle; both primitives must exclude each other
    # on the same path (one descriptor lifecycle, one OS lock — not two lookalikes).
    lock_path = tmp_path / "x.lock"
    handle = FileLockHandle(lock_path)
    with file_lock(lock_path):
        assert not handle.try_acquire()
    assert handle.try_acquire()
    try:
        with pytest.raises(RepositoryError, match="could not acquire"):
            with file_lock(lock_path, timeout=0.2):
                pass
    finally:
        handle.release()
    with file_lock(lock_path, timeout=0.2):
        pass  # released handle frees the path for file_lock again


def test_registry_locked_rereads_from_disk(tmp_path):
    """Two registry instances on the same file: mutations made by one are visible to the
    other's next locked() transaction — no lost updates from stale in-memory state."""
    from seiyuu.voices import CloudVoiceRegistry

    a = CloudVoiceRegistry(tmp_path)
    b = CloudVoiceRegistry(tmp_path)  # loads the (empty) state now
    with a.locked():
        a.touch("v1", "cloud_1")
    with b.locked() as fresh:
        assert fresh.get("v1") == "cloud_1"  # saw a's write despite the stale __init__ load


def test_single_voice_render_gates_cloned_voices(tmp_path):
    """THE bypass the review caught: `render --engine chatterbox --voice <clone>` used to
    synthesize a whole book with no consent check at all. render_book now verifies
    through the library whenever the voice id resolves to a library voice."""
    lib = VoiceLibrary(tmp_path / "voices")

    # a bare voice dir with only reference.wav (never attested) must refuse
    _reference(lib, "stolen_x")
    with pytest.raises(RenderError, match="not found"):
        render_book(
            make_book(), FakeEngine(), "stolen_x", tmp_path / "out1",
            gpu=GpuResourceManager(), library=lib,
        )  # fmt: skip

    # attested but audio swapped afterwards must refuse
    digest = _reference(lib, "elena_x")
    lib.save(_cloned_meta(consent=ConsentAttestation(attested_by="ann", reference_sha256=digest)))
    _reference(lib, "elena_x", fill=0.9)
    with pytest.raises(RenderError, match="consent"):
        render_book(
            make_book(), FakeEngine(), "elena_x", tmp_path / "out2",
            gpu=GpuResourceManager(), library=lib,
        )  # fmt: skip

    # a properly attested clone renders; a bare preset id (no library dir) is untouched
    _reference(lib, "elena_x", fill=0.1)  # restore the attested bytes
    render_book(
        make_book(), FakeEngine(), "elena_x", tmp_path / "out3",
        gpu=GpuResourceManager(), library=lib,
    )  # fmt: skip
    render_book(
        make_book(), FakeEngine(), "af_heart", tmp_path / "out4",
        gpu=GpuResourceManager(), library=lib,
    )  # fmt: skip


def test_cli_single_voice_render_refuses_unattested_clone(tmp_path):
    """End to end through the CLI: the documented bypass invocation now refuses."""
    from click.testing import CliRunner

    from seiyuu.cli import main
    from seiyuu.ingest.epub import write_normalized

    book = make_book()
    write_normalized(book, tmp_path / "books")
    lib = VoiceLibrary(tmp_path / "voices")
    _reference(lib, "stolen_x")  # reference.wav only — never attested

    result = CliRunner().invoke(
        main,
        [
            "render",
            book.book_meta.book_id,
            "--engine",
            "chatterbox",
            "--voice",
            "stolen_x",
            "--books-dir",
            str(tmp_path / "books"),
            "--output-dir",
            str(tmp_path / "output"),
            "--voices-dir",
            str(lib.voices_dir),
        ],  # fmt: skip
    )
    assert result.exit_code != 0
    assert "not found" in result.output
    assert not (tmp_path / "output" / book.book_meta.book_id / "manifest.json").exists()


# --- single-voice render through the GPU manager ---


def test_single_voice_render_uses_and_frees_the_gpu(tmp_path):
    acquired: list[str] = []

    class SpyGpu(GpuResourceManager):
        def acquire(self, consumer, name):
            acquired.append(name)
            return super().acquire(consumer, name)

    gpu = SpyGpu()
    render_book(make_book(), FakeEngine(), "test_voice", tmp_path / "out", gpu=gpu)
    assert acquired and set(acquired) == {"fake"}  # every synth went through the manager
    assert gpu.resident is None  # freed at the end, even on the happy path


def test_single_voice_render_frees_gpu_on_failure(tmp_path):
    gpu = GpuResourceManager()
    with pytest.raises(RenderError):
        render_book(
            make_book(), FakeEngine(fail_on="Hello"), "test_voice", tmp_path / "out", gpu=gpu
        )
    assert gpu.resident is None  # the failure path frees VRAM too


def test_cloud_engine_single_voice_never_touches_the_gpu(tmp_path):
    from test_render_cost_gate import FakeElevenEngine

    class ExplodingGpu(GpuResourceManager):
        def acquire(self, consumer, name):  # pragma: no cover - failing is the assertion
            raise AssertionError("cloud engine must not acquire the GPU")

    render_book(
        make_book(), FakeElevenEngine(), "voice_x", tmp_path / "out",
        gpu=ExplodingGpu(), allow_paid=True, max_paid_usd=5.0,
    )  # fmt: skip
