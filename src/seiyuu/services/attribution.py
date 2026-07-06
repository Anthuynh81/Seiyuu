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
from seiyuu.attribute.aliases import resolve_registry_aliases
from seiyuu.attribute.models import AttributionReport, CharacterRegistry
from seiyuu.attribute.pipeline import _drop_superseded_notes
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


def build_provider(
    cfg, provider_id: str, model: str, prompt_version: str, *, emit_thoughts: bool = False
):
    """Construct an attribution provider, passing only the kwargs each backend needs."""
    from seiyuu.attribute.providers import get_provider

    kwargs = {"prompt_version": prompt_version, "emit_thoughts": emit_thoughts}
    if provider_id == "local":
        kwargs["base_url"] = cfg.ollama_base_url
        kwargs["transport"] = cfg.ollama_transport
        kwargs["num_ctx"] = cfg.ollama_num_ctx
        kwargs["keep_alive"] = cfg.ollama_keep_alive
        kwargs["unload_poll_timeout"] = cfg.gpu_unload_poll_timeout
    elif provider_id == "anthropic":
        kwargs["api_key"] = cfg.anthropic_api_key
    return get_provider(provider_id, model=model, prompts_dir=cfg.prompts_dir, **kwargs)


def _adjudication_model(cfg, provider_id: str) -> str:
    """The adjudication model, defaulting per-provider when unset."""
    if cfg.adjudication_model:
        return cfg.adjudication_model
    return cfg.anthropic_model if provider_id == "anthropic" else cfg.attribution_model


def build_adjudicator(cfg, cache, book_id: str):
    """Construct the cache-wrapped LLM alias adjudicator (the AliasResolver).

    Reuses ``build_provider`` (and thus the anthropic missing-key ctor gate — a PAID
    adjudicator never runs without an explicit key). Returns an ``LLMAdjudicator`` bound to
    the per-book adjudication cache; the caller passes it as ``attribute_book``'s ``resolver``.
    """
    from seiyuu.attribute.adjudicate import LLMAdjudicator

    provider_id = cfg.adjudication_provider
    model = _adjudication_model(cfg, provider_id)
    provider = build_provider(cfg, provider_id, model, cfg.adjudication_prompt_version)
    return LLMAdjudicator(
        provider,
        cache=cache,
        book_id=book_id,
        prompt_version=cfg.adjudication_prompt_version,
        prompts_dir=cfg.prompts_dir,
    )


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
    use_adjudicate: bool | None = None,
    emit_thoughts: bool | None = None,
    chapters: tuple[int, ...] = (),
    progress: Callable[[str], None] | None = None,
    check_cancel: Callable[[], None] | None = None,
    gpu=None,
    provider=None,  # injectable for tests; built from cfg otherwise
    escalation_provider=None,
    resolver=None,  # injectable AliasResolver for tests; built from cfg otherwise
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
    emit_thoughts = cfg.emit_thoughts if emit_thoughts is None else emit_thoughts
    # Thought emission requires the thought-aware v4 prompt; opting in forces it (and thus a
    # distinct cache key) unless the caller pins an explicit --prompt-version.
    default_prompt_version = "v4" if emit_thoughts else cfg.attribution_prompt_version
    prompt_version = prompt_version or default_prompt_version
    use_hybrid = cfg.attribution_hybrid if use_hybrid is None else use_hybrid
    use_adjudicate = cfg.attribution_adjudicate if use_adjudicate is None else use_adjudicate

    provider = provider or build_provider(
        cfg, provider_id, model, prompt_version, emit_thoughts=emit_thoughts
    )
    if escalation_provider is None and use_hybrid and provider_id != "anthropic":
        escalation_provider = build_provider(
            cfg, "anthropic", cfg.anthropic_model, prompt_version, emit_thoughts=emit_thoughts
        )

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
                run_resolver = resolver
                # Adjudication runs only on a FULL-book attribute: a --chapter subset carries a
                # partial registry (merged later by _merge_partial_attribution), and adjudicating
                # an incomplete registry is unsafe. The standalone `adjudicate` command is the
                # registry-complete path for re-runs.
                if run_resolver is None and use_adjudicate and not chapters:
                    run_resolver = build_adjudicator(cfg, cache, book.book_meta.book_id)
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
                    resolver=run_resolver,
                    adjudication_confidence_threshold=cfg.adjudication_confidence_threshold,
                    adjudication_candidate_cap=cfg.adjudication_candidate_cap,
                    adjudication_use_nicknames=cfg.adjudication_use_nicknames,
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


def run_adjudication(
    book_dir: Path,
    *,
    cfg,
    progress: Callable[[str], None] | None = None,
    gpu=None,
    resolver=None,  # injectable AliasResolver for tests; built from cfg otherwise
) -> AttributionReport:
    """Standalone opt-in LLM alias adjudication over the registry-complete RAW report.

    Loads ``attribution.json`` (always the full book), regenerates candidates from its
    registry, runs the cached adjudicator, applies approved merges to the registry and
    segment speakers, and rewrites ``attribution.json``. Idempotent: the per-book cache means
    an unchanged candidate set replays cached verdicts with no LLM call, so the file is
    byte-stable across reruns. The GPU is acquired only when the adjudication provider is
    local (``uses_gpu``) and freed in ``finally``; anthropic (network-only) never touches it.
    """
    say = progress or (lambda _msg: None)
    book_dir = Path(book_dir)
    path = book_dir / ATTRIBUTION_NAME
    if not path.is_file():
        raise ServiceError(f"no attribution at {path}; run `seiyuu attribute` first")
    try:
        report = AttributionReport.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValidationError, OSError) as exc:
        raise ServiceError(
            f"corrupt attribution report {path}: {exc}; re-run `seiyuu attribute`"
        ) from exc

    gpu = gpu or get_gpu_manager()
    with AttributionCache(book_dir / "attribution.db") as cache:
        adjudicator = resolver or build_adjudicator(cfg, cache, report.book_id)
        provider = getattr(adjudicator, "provider", None)
        uses_gpu = bool(provider is not None and getattr(adjudicator, "uses_gpu", False))
        ctx = (
            gpu.acquire(provider, f"llm:adjudicate:{provider.provider_id}:{provider.model_id}")
            if uses_gpu
            else nullcontext()
        )
        registry = report.registry
        pre_names = {c.id: c.canonical_name for c in registry.characters}
        try:
            with ctx:
                id_remap, alias_notes = resolve_registry_aliases(
                    registry,
                    report.chapters,
                    resolver=adjudicator,
                    confidence_threshold=cfg.adjudication_confidence_threshold,
                    candidate_cap=cfg.adjudication_candidate_cap,
                    use_nicknames=cfg.adjudication_use_nicknames,
                )
        finally:
            if uses_gpu:
                gpu.free_all()

    if id_remap:
        for chapter_out in report.chapters:
            chapter_out.segments = [
                seg.model_copy(update={"speaker": id_remap[seg.speaker]})
                if seg.speaker in id_remap
                else seg
                for seg in chapter_out.segments
            ]
    kept = _drop_superseded_notes(report.registry_notes, {pre_names[loser] for loser in id_remap})
    notes = list(dict.fromkeys([*kept, *alias_notes]))
    updated = report.model_copy(update={"registry": registry, "registry_notes": notes})
    write_attribution(updated, book_dir)
    say(f"adjudication: {len(id_remap)} merge(s) applied")
    return updated
