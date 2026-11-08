"""Design a voice from a written description (Qwen3-TTS VoiceDesign), then keep one take.

The flow behind "Describe a voice" (the Voices screen, ``seiyuu voice design``) and the
batch script ``scripts/design_voices.py``:

1. **Design**: the VoiceDesign checkpoint speaks a sample line in a NEW voice made from the
   description, several times with different pinned seeds (takes). Each take is scored with the
   whisper speech check against the line and for pace, and the best is recommended:
   passes the speech check > pace in range > match score > pace nearest the target.
2. **Keep**: the chosen take becomes a library voice: a CLONED ``qwen3tts`` voice whose
   ``reference.wav`` is the take, so every later line is cloned from it with the Base checkpoint.
   A take that passed the speech check also binds its exact text as the clip's transcript
   (ICL mode); a flagged one clones from the sound alone. The consent record attests a
   synthetic voice with no real person behind it, bound to the clip's hash like any clone.

A design lives in ``{data_dir}/designs/{design_id}/`` (``design.json`` + ``take{k}.wav``) until
it is deleted; keeping a take copies it into the library, so the two never share a file.
"""

import hashlib
import os
import re
import secrets
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from seiyuu.repository import atomic_write_text
from seiyuu.services.clone_transcript import bind_transcript
from seiyuu.services.common import ServiceError
from seiyuu.voices import VoiceLibrary
from seiyuu.voices.library import sha256_file
from seiyuu.voices.models import ConsentAttestation, VoiceKind, VoiceMeta

DESIGN_ENGINE = "qwen3tts"
SOURCE = "qwen_design"
CONSENT_TEXT = "synthetic voice designed from a text description, no real person"
ATTESTED_BY = "Seiyuu voice design (Qwen3-TTS VoiceDesign)"
WPM_RANGE = (140.0, 210.0)
WPM_TARGET = 175.0
MAX_TAKES = 5
DEFAULT_TAKES = 3
# Long enough to judge a voice and to clone from (~8-10 s), plain enough to read in any genre.
DEFAULT_SAMPLE_TEXT = (
    "I didn't expect to find anyone here at this hour. Still, since you've come all this way, "
    "you might as well sit down and tell me what happened."
)
RECORD_NAME = "design.json"
DESIGN_ID_RE = re.compile(r"^[0-9a-f]{12}$")


@dataclass
class TakeScore:
    take: int
    seed: int
    path: str
    text: str  # the normalized text the model spoke (also the reference transcript)
    seconds: float
    words: int
    wpm: float
    ok: bool
    score: float
    transcript: str
    reason: str | None = None


def take_seed(key: str, take: int) -> int:
    """A stable, distinct seed per (voice or design, take): re-runs reproduce the same takes."""
    digest = hashlib.sha256(f"{key}#{take}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % 2_000_000_000


def words_per_minute(text: str, seconds: float) -> float:
    return len(text.split()) / max(seconds, 1e-6) * 60.0


def in_pace(wpm: float) -> bool:
    return WPM_RANGE[0] <= wpm <= WPM_RANGE[1]


def rank_key(t: TakeScore) -> tuple:
    return (t.ok, in_pace(t.wpm), round(t.score, 2), -abs(t.wpm - WPM_TARGET), -t.take)


def pick_best(takes: list[TakeScore]) -> TakeScore:
    if not takes:
        raise ValueError("no takes to pick from")
    return max(takes, key=rank_key)


def score_take(validator: Any, path: Path, text: str, take: int, seed: int) -> TakeScore:
    import soundfile as sf

    info = sf.info(str(path))
    seconds = info.frames / float(info.samplerate)
    verdict = validator.validate(path, text)
    return TakeScore(
        take=take,
        seed=seed,
        path=str(path),
        text=text,
        seconds=round(seconds, 2),
        words=len(text.split()),
        wpm=round(words_per_minute(text, seconds), 1),
        ok=bool(verdict.ok),
        score=float(verdict.score),
        transcript=verdict.transcript,
        reason=verdict.reason,
    )


# -- a design on disk ---------------------------------------------------------------------------


class DesignTake(BaseModel):
    take: int
    seed: int
    seconds: float
    wpm: float
    ok: bool  # passed the speech check
    score: float
    transcript: str
    reason: str | None = None
    in_pace: bool


class DesignRecord(BaseModel):
    """``design.json``: what was asked, the takes so far, and which one is recommended."""

    design_id: str
    name: str
    description: str
    text: str  # the line every take speaks, normalized for qwen3tts
    takes_requested: int = Field(ge=1, le=MAX_TAKES)
    takes: list[DesignTake] = []
    recommended: int | None = None
    kept: dict[int, str] = {}  # take -> voice_id it became


def new_design_id() -> str:
    return secrets.token_hex(6)


def design_dir(data_dir: Path, design_id: str) -> Path:
    if not DESIGN_ID_RE.match(design_id):
        raise ServiceError(f"invalid design id {design_id!r}")
    return Path(data_dir) / "designs" / design_id


def take_path(directory: Path, take: int) -> Path:
    return directory / f"take{take}.wav"


def save_record(directory: Path, record: DesignRecord) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write_text(directory / RECORD_NAME, record.model_dump_json(indent=2))


def load_record(directory: Path) -> DesignRecord:
    path = directory / RECORD_NAME
    if not path.is_file():
        raise ServiceError(f"no voice design at {directory}")
    return DesignRecord.model_validate_json(path.read_text(encoding="utf-8"))


def start_design(
    data_dir: Path, *, name: str, description: str, text: str, takes: int
) -> DesignRecord:
    """Write the request as a fresh design (no takes yet) and return it."""
    if not description.strip():
        raise ServiceError("a voice design needs a description")
    if not text.strip():
        raise ServiceError("a voice design needs a sample line to speak")
    design_id = new_design_id()
    record = DesignRecord(
        design_id=design_id,
        name=name.strip() or "Designed voice",
        description=description.strip(),
        text=text.strip(),
        takes_requested=takes,
    )
    save_record(design_dir(data_dir, design_id), record)
    return record


def run_design(
    directory: Path,
    *,
    engine: Any,
    validator: Any,
    gpu: Any,
    progress: Callable[[str], None] = lambda _msg: None,
    check_cancel: Callable[[], None] = lambda: None,
    release: bool = True,
) -> DesignRecord:
    """Design every requested take not yet on disk, scoring each, and recommend the best.

    ``release=False`` leaves the VoiceDesign checkpoint resident for a caller designing several
    voices in a row (the design-cast job); that caller frees the GPU itself when it is done.

    ``engine`` needs ``design_voice``; ``gpu`` is the GpuResourceManager (``acquire`` /
    ``free_all``). The record is saved after every take, so a canceled or failed run keeps what
    it made, and a rerun designs only the missing takes (seeds are pinned per design + take).
    The VoiceDesign checkpoint is unloaded at the end: nothing renders with it afterwards."""
    record = load_record(directory)
    done = {t.take for t in record.takes}
    acquired = False
    try:
        for k in range(1, record.takes_requested + 1):
            if k in done:
                continue
            check_cancel()
            progress(f"designing take {k} of {record.takes_requested}")
            seed = take_seed(record.design_id, k)
            with gpu.acquire(engine, f"{DESIGN_ENGINE}:design"):
                acquired = True
                audio = engine.design_voice(record.text, record.description, seed=seed)
            path = take_path(directory, k)
            audio.save(path)
            progress(f"checking take {k} of {record.takes_requested}")
            scored = score_take(validator, path, record.text, k, seed)
            record.takes.append(
                DesignTake(
                    take=k,
                    seed=seed,
                    seconds=scored.seconds,
                    wpm=scored.wpm,
                    ok=scored.ok,
                    score=scored.score,
                    transcript=scored.transcript,
                    reason=scored.reason,
                    in_pace=in_pace(scored.wpm),
                )
            )
            record.takes.sort(key=lambda t: t.take)
            record.recommended = pick_best(
                [_as_score(directory, record, t) for t in record.takes]
            ).take
            save_record(directory, record)
    finally:
        if acquired and release:
            gpu.free_all()
    return record


def _as_score(directory: Path, record: DesignRecord, t: DesignTake) -> TakeScore:
    return TakeScore(
        take=t.take,
        seed=t.seed,
        path=str(take_path(directory, t.take)),
        text=record.text,
        seconds=t.seconds,
        words=len(record.text.split()),
        wpm=t.wpm,
        ok=t.ok,
        score=t.score,
        transcript=t.transcript,
        reason=t.reason,
    )


def keep_take(
    directory: Path,
    take: int,
    *,
    library: VoiceLibrary,
    name: str | None = None,
    voice_id: str | None = None,
    tags: list[str] | None = None,
) -> VoiceMeta:
    """Make take ``take`` of the design a library voice; returns its meta.

    Copy, hash the COPY, publish atomically, then save the meta (the clone ordering), so the
    consent record describes exactly the bytes the library serves. Never overwrites a voice."""
    record = load_record(directory)
    chosen = next((t for t in record.takes if t.take == take), None)
    src = take_path(directory, take)
    if chosen is None or not src.is_file():
        raise ServiceError(f"design {record.design_id} has no take {take}")
    vid = voice_id or library.new_voice_id(name or record.name)
    if library.meta_path(vid).is_file() or library.reference_path(vid).is_file():
        raise ServiceError(f"voice {vid!r} already exists; pick another name")
    ref = library.reference_path(vid)
    ref.parent.mkdir(parents=True, exist_ok=True)
    tmp = ref.with_name("reference.wav.part")
    shutil.copyfile(src, tmp)
    sha = sha256_file(tmp)
    os.replace(tmp, ref)
    # a take that passed the speech check said exactly record.text: bind it (ICL mode)
    settings = bind_transcript(ref, sha, record.text if chosen.ok else None)
    meta = VoiceMeta(
        voice_id=vid,
        name=(name or record.name).strip(),
        kind=VoiceKind.CLONED,
        engine=DESIGN_ENGINE,
        reference_audio="reference.wav",
        seed=chosen.seed,
        consent_attested=True,
        consent=ConsentAttestation(
            attested_by=ATTESTED_BY, statement=CONSENT_TEXT, reference_sha256=sha
        ),
        source=SOURCE,
        tags=tags or ["designed"],
        settings=settings,
    )
    library.save(meta)
    record.kept[take] = vid
    save_record(directory, record)
    return meta
