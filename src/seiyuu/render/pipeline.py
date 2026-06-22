"""Single-voice render: normalized JSON → cached canonical segment WAVs + manifest.

Only speakable blocks (paragraph, heading) become synthesis segments; scene
breaks pass through to the manifest as pause markers. Every synthesis call
goes through the segment cache.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile as sf

from seiyuu.engines import TTSEngine
from seiyuu.ingest.models import BlockType, NormalizedBook
from seiyuu.normalize import normalize_text
from seiyuu.render.cache import SegmentCache, SegmentKey
from seiyuu.render.models import RenderedChapter, RenderedSegment, RenderManifest

MANIFEST_NAME = "manifest.json"


class RenderError(Exception):
    """Loud render failure naming book/chapter/block."""


@dataclass
class RenderResult:
    manifest: RenderManifest
    manifest_path: Path
    synthesized: int
    cache_hits: int

    @property
    def total_audio_seconds(self) -> float:
        return sum(s.duration_seconds for c in self.manifest.chapters for s in c.segments)


def render_book(
    book: NormalizedBook,
    engine: TTSEngine,
    voice_id: str,
    book_output_dir: Path,
    *,
    settings: dict[str, Any] | None = None,
    seed: int | None = None,
    chapters: tuple[int, ...] = (),
    progress: Callable[[str], None] | None = None,
) -> RenderResult:
    """Render a book (or a 1-based subset of `chapters`) with one voice."""
    settings = settings or {}
    book_output_dir = Path(book_output_dir)
    cache = SegmentCache(book_output_dir / "cache")
    say = progress or (lambda _msg: None)

    wanted = set(chapters)
    unknown = wanted - set(range(1, len(book.chapters) + 1))
    if unknown:
        raise RenderError(
            f"{book.book_meta.book_id}: no such chapter(s) {sorted(unknown)} "
            f"(book has {len(book.chapters)})"
        )

    rendered_chapters: list[RenderedChapter] = []
    synthesized = cache_hits = 0
    for ci, chapter in enumerate(book.chapters, start=1):
        if wanted and ci not in wanted:
            continue
        say(f"chapter {ci}/{len(book.chapters)}: {chapter.title}")
        segments: list[RenderedSegment] = []
        for block in chapter.blocks:
            if block.type is BlockType.SCENE_BREAK:
                segments.append(RenderedSegment(block_id=block.id, type=block.type))
                continue
            text = normalize_text(block.text)
            key = SegmentKey.build(
                engine=engine.engine_id,
                engine_model_version=engine.model_version,
                voice_id=voice_id,
                settings=settings,
                seed=seed,
                normalized_text=text,
            )
            wav_path = cache.get(key)
            if wav_path is not None:
                cache_hits += 1
                duration = sf.info(str(wav_path)).duration
            else:
                try:
                    audio = engine.synthesize(
                        text, voice_id, settings | ({"seed": seed} if seed is not None else {})
                    )
                except Exception as exc:
                    raise RenderError(
                        f"synthesis failed: book={book.book_meta.book_id} "
                        f"chapter={ci} ({chapter.title!r}) block={block.id} "
                        f"engine={engine.engine_id} voice={voice_id}: {exc}"
                    ) from exc
                wav_path = cache.put(key, audio)
                synthesized += 1
                duration = audio.duration_seconds
            segments.append(
                RenderedSegment(
                    block_id=block.id,
                    type=block.type,
                    wav=wav_path.relative_to(book_output_dir).as_posix(),
                    duration_seconds=round(duration, 3),
                )
            )
        rendered_chapters.append(RenderedChapter(index=ci, title=chapter.title, segments=segments))

    manifest = RenderManifest(
        book_id=book.book_meta.book_id,
        book_title=book.book_meta.title,
        engine=engine.engine_id,
        engine_model_version=engine.model_version,
        voice_id=voice_id,
        settings=settings,
        seed=seed,
        chapters=rendered_chapters,
    )
    manifest_path = book_output_dir / MANIFEST_NAME
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return RenderResult(
        manifest=manifest,
        manifest_path=manifest_path,
        synthesized=synthesized,
        cache_hits=cache_hits,
    )
