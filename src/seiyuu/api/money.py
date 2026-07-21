"""Server-side cost estimation shared by the estimate/quote routes, the render
enqueue dry-run, and the render job handler.

One code path builds the estimate for all four callers — the quote fingerprint binds to
the exact paid SegmentKeys, so the enqueue-time dry-run, the handler's consume-time
verify, and what the user approved must all be computed identically or verification
would refuse spuriously (or worse, pass wrongly). Single-voice defaults (engine, voice)
resolve HERE, once. Estimates are pure reads: no network, no GPU, no signing-key state.
"""

from dataclasses import dataclass
from pathlib import Path

from seiyuu.api.registry import EngineRegistry
from seiyuu.api.schemas import SingleSpec
from seiyuu.ingest.models import NormalizedBook
from seiyuu.normalize.lexicon import load_compiled_lexicon
from seiyuu.render.gate import CostGateError, hash_assignment
from seiyuu.render.pipeline import (
    CostEstimate,
    estimate_render_cost,
    estimate_render_cost_single,
)
from seiyuu.services import load_assignment, load_report
from seiyuu.settings import Settings
from seiyuu.voices import VoiceLibrary


@dataclass(frozen=True)
class ResolvedSingle:
    """The single-voice spec with settings defaults applied — resolved once, used by
    every path (estimate, quote, dry-run, handler) so SegmentKeys always agree."""

    engine_id: str
    voice_id: str
    speed: float
    seed: int

    @property
    def settings(self) -> dict:
        return {"speed": self.speed}


def effective_apply_emotion(cfg: Settings, override: bool | None) -> bool:
    """F2b: the per-render apply_emotion override resolved against the server default.

    ``None`` -> ``cfg.apply_emotion`` (the server-global default); an explicit bool wins. The
    render handler and the cost estimate MUST both resolve through this one helper so the gate
    authorizes exactly the emotion-folded SegmentKeys render bills (parity)."""
    return cfg.apply_emotion if override is None else override


def resolve_single(cfg: Settings, spec: SingleSpec | None) -> ResolvedSingle:
    spec = spec or SingleSpec()
    engine_id = spec.engine or cfg.tts_engine
    voice_id = spec.voice
    if voice_id is None:
        # The settings default is a KOKORO preset; silently handing it to another
        # engine would mint a real-dollar quote (and later burn its token) for a
        # render that can never synthesize. Routes map this to a 422.
        if engine_id != "kokoro":
            raise ValueError(
                f"single-voice render with engine {engine_id!r} requires an explicit voice"
            )
        voice_id = cfg.kokoro_default_voice
    return ResolvedSingle(
        engine_id=engine_id,
        voice_id=voice_id,
        speed=spec.speed,
        seed=spec.seed,
    )


@dataclass(frozen=True)
class EstimateContext:
    est: CostEstimate
    assignment_hash: str | None  # multivoice binding; None for single-voice
    edit_warnings: list[str]


def compute_estimate(
    cfg: Settings,
    registry: EngineRegistry,
    book: NormalizedBook,
    book_id: str,
    *,
    mode: str,
    chapters: tuple[int, ...],
    single: ResolvedSingle | None,
    apply_emotion: bool | None = None,
    force: bool = False,
) -> EstimateContext:
    """The fresh estimate for a render exactly as the render loop would bill it.

    Raises the underlying loud errors (ServiceError for missing/corrupt artifacts,
    VoiceLibraryError for unknown voices, ValueError for unknown engines) — callers map
    them to their boundary. Multivoice estimates use the EFFECTIVE report, so a manual
    edit between estimate and render correctly drifts the fingerprint and refuses.

    ``apply_emotion`` (F2b) is the per-render override; ``None`` falls back to
    ``cfg.apply_emotion``. Resolved via ``effective_apply_emotion`` so estimate and render
    build the IDENTICAL emotion-folded keys.
    """
    book_output_dir = Path(cfg.output_dir) / book_id
    library = VoiceLibrary(cfg.voices_dir)
    # F3: the SAME compiled per-book lexicon the render loop uses, so the cost gate authorizes
    # exactly the SegmentKeys render will bill. Loaded once here; render loads the same file.
    lexicon = load_compiled_lexicon(cfg.books_dir / book_id)
    if mode == "multivoice":
        report, warnings = load_report(cfg.books_dir / book_id)
        assignment = load_assignment(cfg.output_dir, book_id)
        est = estimate_render_cost(
            report, book, library, assignment, book_output_dir,
            chapters=chapters, lexicon=lexicon,
            # F2/F2b: the SAME effective flag the render handler passes to render_book_multivoice
            # (per-render override, else the cfg default), so the cost gate authorizes exactly the
            # SegmentKeys render will bill.
            apply_emotion=effective_apply_emotion(cfg, apply_emotion),
            # force must match the render's force so the priced paid set equals what render bills.
            force=force,
        )  # fmt: skip
        return EstimateContext(
            est=est, assignment_hash=hash_assignment(assignment), edit_warnings=warnings
        )
    assert single is not None  # RenderParams/QuoteRequest validators enforce this
    engine = registry.get(single.engine_id)  # shared instance: same model_version everywhere
    est = estimate_render_cost_single(
        book,
        engine,
        single.voice_id,
        book_output_dir,
        settings=single.settings,
        seed=single.seed,
        chapters=chapters,
        library=library,
        lexicon=lexicon,
        force=force,
    )
    return EstimateContext(est=est, assignment_hash=None, edit_warnings=[])


# CostGateError message -> the granular 402 code (scoping doc section 1). The messages
# are the gate's stable, tested contract; M6c branches on the code to decide between
# "re-mint silently" and "re-show the cost dialog". Needles are FULL multi-word phrases:
# the book-mismatch message interpolates user-controlled book ids (slugs like
# "the-expired-heart-1a2b"), so a bare "expired"/"ceiling" needle would cross-map a
# mismatch to the wrong code — and mismatch needles check first as defense in depth.
_GATE_CODES = (
    ("issued for book", "quote_mismatch"),
    ("chapter selection", "quote_mismatch"),
    ("assignment changed", "quote_mismatch"),
    ("paid segments changed", "quote_mismatch"),
    ("already used", "quote_used"),
    ("cost token expired", "quote_expired"),
    ("signature invalid", "quote_signature_invalid"),
    ("render_max_usd ceiling", "ceiling_exceeded"),
    ("drifted upward", "cost_drift"),
)


def gate_code(exc: CostGateError) -> str:
    message = str(exc)
    for needle, code in _GATE_CODES:
        if needle in message:
            return code
    return "quote_refused"
