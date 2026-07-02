"""Attribution service: the runnable shared by the CLI, job handlers, and the API.

Owns the full lifecycle the CLI used to hand-roll: provider construction, GPU manager
acquisition, the chunk cache, writing the RAW attribution.json, and applying the
manual-edits overlay. Two invariants live here so no caller can forget them:

- The local LLM is a heavy GPU consumer like any TTS engine, so the run holds the GPU
  manager for its duration — in a server, an attribution job must prevent a concurrent
  audition request from loading a TTS model alongside Ollama on the single card.
- Ollama VRAM is freed in a ``finally`` (via the manager, keeping its residency state
  truthful), so a canceled or failed job can never leave the LLM resident.

``attribution.json`` always stores the RAW report; every reader goes through
``load_report`` so manual edits are replayed uniformly (characters overview, assignment
drafting, cost estimation, and the multi-voice render all see the same effective report).
"""

from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path

from pydantic import ValidationError

from seiyuu.attribute import ATTRIBUTION_NAME, AttributionCache, attribute_book, write_attribution
from seiyuu.attribute.models import AttributionReport, CharacterRegistry
from seiyuu.gpu import get_gpu_manager
from seiyuu.ingest.models import NormalizedBook
from seiyuu.repository import file_lock
from seiyuu.services.common import ServiceError
from seiyuu.services.edits import (
    EditOp,
    anchor_op,
    apply_edits,
    edits_path,
    load_edits,
    save_edits,
)


def build_provider(cfg, provider_id: str, model: str, prompt_version: str):
    """Construct an attribution provider, passing only the kwargs each backend needs."""
    from seiyuu.attribute.providers import get_provider

    kwargs = {"prompt_version": prompt_version}
    if provider_id == "local":
        kwargs["base_url"] = cfg.ollama_base_url
        kwargs["transport"] = cfg.ollama_transport
        kwargs["num_ctx"] = cfg.ollama_num_ctx
        kwargs["keep_alive"] = cfg.ollama_keep_alive
        kwargs["unload_poll_timeout"] = cfg.gpu_unload_poll_timeout
    elif provider_id == "anthropic":
        kwargs["api_key"] = cfg.anthropic_api_key
    return get_provider(provider_id, model=model, prompts_dir=cfg.prompts_dir, **kwargs)


def load_report(book_dir: Path) -> tuple[AttributionReport, list[str]]:
    """attribution.json + the manual-edits overlay → (effective report, edit warnings)."""
    path = Path(book_dir) / ATTRIBUTION_NAME
    if not path.is_file():
        raise ServiceError(f"no attribution at {path}; run `seiyuu attribute` first")
    try:
        raw = AttributionReport.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValidationError, OSError) as exc:
        raise ServiceError(
            f"corrupt attribution report {path}: {exc}; re-run `seiyuu attribute`"
        ) from exc
    return apply_edits(raw, load_edits(book_dir))


def record_edit(book_dir: Path, op: EditOp) -> EditOp:
    """Validate an edit against the CURRENT effective report, fill its content anchors,
    and append it durably — all under a cross-process lock so two concurrent editors
    (M6b requests) can never lose each other's ops. Raises ``ServiceError`` when the op
    doesn't apply cleanly right now (unknown character/block, index out of range)."""
    book_dir = Path(book_dir)
    with file_lock(edits_path(book_dir).with_name("edits.json.lock")):
        report, _ = load_report(book_dir)
        anchored = anchor_op(report, op)
        log = load_edits(book_dir)
        log.ops.append(anchored)
        save_edits(book_dir, log)
    return anchored


def undo_edit(book_dir: Path) -> EditOp | None:
    """Remove and return the most recent op (None if empty), under the same lock."""
    book_dir = Path(book_dir)
    with file_lock(edits_path(book_dir).with_name("edits.json.lock")):
        log = load_edits(book_dir)
        if not log.ops:
            return None
        op = log.ops.pop()
        save_edits(book_dir, log)
        return op


def _merge_partial_attribution(book_dir: Path, new: AttributionReport) -> AttributionReport:
    """A --chapter subset run must not throw away the rest of the book: the new chapters
    replace their old versions and everything else is kept (registry union with the new
    run winning per id; flags for re-attributed chapters dropped; provenance = last run).
    Without this, `attribute --chapter 2` on a fully-attributed book silently shrank the
    effective book to one chapter for every downstream reader."""
    path = Path(book_dir) / ATTRIBUTION_NAME
    if not path.is_file():
        return new
    old = AttributionReport.model_validate_json(path.read_text(encoding="utf-8"))
    new_indices = {c.index for c in new.chapters}
    by_index = {c.index: c for c in old.chapters} | {c.index: c for c in new.chapters}
    merged_registry = {c.id: c for c in old.registry.characters} | {
        c.id: c for c in new.registry.characters
    }
    return new.model_copy(
        update={
            "chapters": [by_index[i] for i in sorted(by_index)],
            "registry": CharacterRegistry(characters=list(merged_registry.values())),
            "flagged": [f for f in old.flagged if f.chapter_index not in new_indices]
            + list(new.flagged),
            "registry_notes": list(dict.fromkeys([*old.registry_notes, *new.registry_notes])),
        }
    )


def run_attribution(
    book: NormalizedBook,
    book_dir: Path,
    *,
    cfg,
    provider_id: str | None = None,
    model: str | None = None,
    prompt_version: str | None = None,
    use_hybrid: bool | None = None,
    chapters: tuple[int, ...] = (),
    progress: Callable[[str], None] | None = None,
    check_cancel: Callable[[], None] | None = None,
    gpu=None,
    provider=None,  # injectable for tests; built from cfg otherwise
    escalation_provider=None,
) -> AttributionReport:
    """Attribute ``book``: cache-aware LLM run → RAW attribution.json → effective report.

    Returns the EFFECTIVE report (manual edits applied); edit warnings go to
    ``progress``. The GPU manager is held for the whole run and freed in ``finally``.
    """
    say = progress or (lambda _msg: None)
    book_dir = Path(book_dir)
    gpu = gpu or get_gpu_manager()
    provider_id = provider_id or cfg.attribution_provider
    model = model or cfg.attribution_model
    prompt_version = prompt_version or cfg.attribution_prompt_version
    use_hybrid = cfg.attribution_hybrid if use_hybrid is None else use_hybrid

    provider = provider or build_provider(cfg, provider_id, model, prompt_version)
    if escalation_provider is None and use_hybrid and provider_id != "anthropic":
        escalation_provider = build_provider(cfg, "anthropic", cfg.anthropic_model, prompt_version)

    # cloud providers (anthropic) never touch the card: acquiring would needlessly evict
    # a resident TTS model and serialize the whole M6b server behind a network-only run
    provider_uses_gpu = getattr(provider, "uses_gpu", True)
    ctx = (
        gpu.acquire(provider, f"llm:{provider.provider_id}:{provider.model_id}")
        if provider_uses_gpu
        else nullcontext()
    )
    try:
        with ctx:
            with AttributionCache(book_dir / "attribution.db") as cache:
                raw = attribute_book(
                    book,
                    provider,
                    cache=cache,
                    budget_tokens=cfg.attribution_chunk_tokens,
                    overlap_blocks=cfg.attribution_chunk_overlap_blocks,
                    max_local_retries=cfg.attribution_max_local_retries,
                    escalation_provider=escalation_provider,
                    chapters=chapters,
                    progress=progress,
                    check_cancel=check_cancel,
                )
        if chapters:
            raw = _merge_partial_attribution(book_dir, raw)
        write_attribution(raw, book_dir)
    finally:
        if provider_uses_gpu:
            # frees Ollama VRAM before any TTS engine loads (GPU discipline), even on
            # cancel/failure; free_all() keeps the manager's residency state truthful
            gpu.free_all()

    effective, warnings = apply_edits(raw, load_edits(book_dir))
    for warning in warnings:
        say(f"edit overlay: {warning}")
    return effective
