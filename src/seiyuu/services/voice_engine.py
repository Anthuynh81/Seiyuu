"""Switch a cloned voice to another local cloning engine, in place (decision D2, 2026-11-09).

A cloned voice is a reference clip, its consent record and (for some engines) the clip's
transcript. None of that belongs to an engine, so the engine that clones it can change while
the voice keeps its id, and every book's casting keeps pointing at it.

What makes this safe:

- the engine is part of every segment's cache key, so audio made on the old engine can never
  be served as the new engine's (and it is still there if the voice is switched back);
- consent binds to the clip's bytes, which do not change;
- nothing is deleted: cached audio and manifests stay valid for the engine that made them.

What a switch does to ``meta.json``: ``engine`` changes, and the transcript binding
(``ref_text_digest``) is COPIED under the new engine's settings when that engine clones from a
transcript. Copied, not moved: the old engine's block stays, so a switch back builds the old
cache keys again. Sampler tunables are never copied (they would change the new engine's keys
for nothing).

Breeze cannot clone without a transcript. A voice that binds none (a Chatterbox or IndexTTS-2
clone, a Qwen clone of a clip over 30 s) gets one from whisper on the CPU, or the switch is
refused; it is never saved as a voice that fails on its first line.

Only cloned voices, only between engines that clone locally from the library's clip. Books
rendered with the voice are NOT marked stale by this (the casting did not change): the caller
is told which books use the voice so the user can re-record them.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from seiyuu.engines import get_engine_class, list_engine_ids
from seiyuu.engines.breeze_engine import breeze_ready
from seiyuu.engines.qwen3tts_engine import REF_TEXT_DIGEST_KEY, reference_text_digest
from seiyuu.repository import atomic_write_text
from seiyuu.services.clone_transcript import (
    TRANSCRIPT_ENGINES,
    TRANSCRIPT_NAME,
    TRANSCRIPT_REQUIRED,
    CloneRefused,
    Transcriber,
    choose_transcript,
)
from seiyuu.services.common import ServiceError
from seiyuu.services.voices import voice_references
from seiyuu.voices import VoiceKind, VoiceLibrary, VoiceLibraryError, VoiceMeta
from seiyuu.voices.library import sha256_file

AUDITION_NAME = "audition.wav"


class EngineNotInstalled(ServiceError):
    """The target engine is not set up on this machine (a switch would break every estimate
    and render of the books that use the voice)."""


@dataclass(frozen=True)
class EngineSwitch:
    meta: VoiceMeta
    previous_engine: str
    note: str  # one plain sentence: what changed, and what the user should do next
    books: list[str]  # book ids whose casting uses this voice (they need re-recording)


def switchable_engines() -> list[str]:
    """Engines that clone locally from the library's reference clip."""
    return [
        engine_id
        for engine_id in list_engine_ids()
        if getattr(get_engine_class(engine_id), "clones_from_library", False)
    ]


def engine_installed(cfg: Any, engine_id: str) -> bool | None:
    """Is a worker engine set up on this machine? None when it cannot be told cheaply (an
    engine that downloads its own weights on first use)."""
    if engine_id == "breeze":
        return breeze_ready(cfg)
    if engine_id == "qwen3tts":
        from seiyuu.engines.qwen3tts_engine import models_present, qwen3tts_config

        return models_present(qwen3tts_config(cfg)["models_dir"])
    if engine_id == "indextts2":
        from seiyuu.engines.indextts2_engine import checkpoints_present

        try:
            return checkpoints_present(getattr(cfg, "indextts2_checkpoints_dir", None))
        except OSError:
            return None
    return None


def _reference_sha(meta: VoiceMeta, reference: Path) -> str:
    """The clip's hash: the consent record's when it has one (a grandfathered clone has not)."""
    recorded = meta.consent.reference_sha256 if meta.consent is not None else None
    return recorded or sha256_file(reference)


def _bound_digest(meta: VoiceMeta, reference: Path) -> float | None:
    """The transcript binding the voice already carries under ANY engine, checked against the
    clip and ``reference.txt`` on disk. None when it binds nothing; a binding that no longer
    matches the files is an error (hand-edited transcript, replaced clip)."""
    bound = next(
        (
            block[REF_TEXT_DIGEST_KEY]
            for block in meta.settings.values()
            if REF_TEXT_DIGEST_KEY in block
        ),
        None,
    )
    if bound is None:
        return None
    path = reference.with_name(TRANSCRIPT_NAME)
    text = path.read_text(encoding="utf-8").strip() if path.is_file() else ""
    if not text or reference_text_digest(_reference_sha(meta, reference), text) != float(bound):
        raise ServiceError(
            f"voice {meta.voice_id!r}: {TRANSCRIPT_NAME} is missing or no longer matches the "
            "transcript its meta binds (edited by hand, or the clip changed); re-clone the voice"
        )
    return float(bound)


def switch_engine(
    library: VoiceLibrary,
    voice_id: str,
    target: str,
    *,
    cfg: Any,
    output_dir: Path,
    transcribe: Transcriber | None = None,
    installed: Callable[[Any, str], bool | None] | None = None,
) -> EngineSwitch:
    """Make ``voice_id`` clone on ``target`` from now on. See the module docstring."""
    try:
        meta = library.load(voice_id)
    except VoiceLibraryError as exc:
        raise ServiceError(str(exc)) from exc
    allowed = switchable_engines()
    if meta.kind is not VoiceKind.CLONED:
        raise ServiceError(
            f"voice {voice_id!r} is a {meta.kind.value} voice; only a cloned voice can move to "
            "another engine"
        )
    if target not in allowed:
        raise ServiceError(
            f"{target!r} is not a local cloning engine; choose one of {', '.join(allowed)}"
        )
    if meta.engine not in allowed:
        raise ServiceError(
            f"voice {voice_id!r} is cloned on {meta.engine}, which keeps its voice in the "
            "cloud; clone it again on a local engine instead"
        )
    if meta.engine == target:
        raise ServiceError(f"voice {voice_id!r} already uses {target}")
    if (installed or engine_installed)(cfg, target) is False:
        raise EngineNotInstalled(
            f"{target} is not set up on this machine, so every book that uses {voice_id!r} "
            "would stop rendering; set the engine up first (see the README)"
        )
    try:
        library.verify_consent(meta)  # also proves the clip is there and is the attested one
    except VoiceLibraryError as exc:
        raise ServiceError(str(exc)) from exc

    reference = library.reference_path(voice_id)
    settings = {engine: dict(block) for engine, block in meta.settings.items()}
    how = ""
    if target in TRANSCRIPT_ENGINES:
        digest = _bound_digest(meta, reference)
        if digest is None and target in TRANSCRIPT_REQUIRED:
            if transcribe is None:
                raise ServiceError(
                    f"voice {voice_id!r} binds no transcript and {target} cannot clone without one"
                )
            try:
                chosen = choose_transcript(
                    reference, provided=None, transcribe=transcribe, engine=target
                )
            except CloneRefused as exc:
                raise ServiceError(f"voice {voice_id!r} cannot move to {target}: {exc}") from exc
            assert chosen.text is not None  # a transcript-required engine never returns None
            atomic_write_text(reference.with_name(TRANSCRIPT_NAME), chosen.text + "\n")
            digest = reference_text_digest(_reference_sha(meta, reference), chosen.text)
            how = " Its clip was transcribed for this (check reference.txt if a line sounds off)."
        if digest is not None:
            settings[target] = {**settings.get(target, {}), REF_TEXT_DIGEST_KEY: digest}
    previous = meta.engine
    meta.engine = target
    meta.settings = settings
    library.save(meta)
    # the saved sample was spoken by the old engine: drop it so the card never plays it as new
    (library.dir_for(voice_id) / AUDITION_NAME).unlink(missing_ok=True)
    books = sorted({ref.book_id for ref in voice_references(voice_id, output_dir)})
    if books:
        again = f" Record {len(books)} book(s) again to hear it: {', '.join(books)}."
    else:
        again = " No book uses it yet."
    note = (
        f"{voice_id} now clones on {target} (was {previous}).{how}{again} Lines already "
        f"recorded on {previous} are kept."
    )
    return EngineSwitch(meta=meta, previous_engine=previous, note=note, books=books)
