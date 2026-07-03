"""FastAPI app factory. Run: ``uvicorn seiyuu.api.main:app`` (exactly ONE worker).

The lifespan owns the singletons, in scoping-doc order: Settings -> JobStore ->
EngineRegistry -> HeavyWorkGate -> handler map -> JobRunner. ``runner.start()``
reconciles the store BEFORE any request is served, so the UI never sees a ghost job.
``--reload`` and ``workers>1`` are unsupported outside development — a duplicate
process would reconcile-kill live job rows and break every process-local primitive
(GPU manager lock, runner queue, heavy-work gate).
"""

import logging
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from seiyuu import __version__
from seiyuu.api.concurrency import HeavyWorkGate
from seiyuu.api.errors import register_error_handlers
from seiyuu.api.handlers import build_handlers
from seiyuu.api.registry import EngineRegistry
from seiyuu.api.routes import books as books_routes
from seiyuu.api.routes import engines as engines_routes
from seiyuu.api.routes import jobs as jobs_routes
from seiyuu.api.routes import render as render_routes
from seiyuu.api.routes import review as review_routes
from seiyuu.api.routes import system as system_routes
from seiyuu.api.routes import voices as voices_routes
from seiyuu.gpu import get_gpu_manager
from seiyuu.jobs import JobRunner
from seiyuu.repository import JobStore
from seiyuu.repository.jobs import JOBS_DB_NAME
from seiyuu.settings import Settings, get_settings

logger = logging.getLogger(__name__)

_SHUTDOWN_TIMEOUT_SECONDS = 10.0


def create_app(*, settings: Settings | None = None) -> FastAPI:
    """App factory. ``settings`` is injectable for tests; production resolves the
    process-wide singleton lazily at startup, not at import."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        cfg = settings or get_settings()
        store = JobStore(cfg.data_dir / JOBS_DB_NAME)
        registry = EngineRegistry(cfg)
        gate = HeavyWorkGate()
        runner = JobRunner(store, build_handlers(cfg, registry, gate))
        reconciled = runner.start()  # reconcile FIRST — before any request is served
        if reconciled:
            logger.info("startup reconcile settled %d orphaned job row(s)", reconciled)
        app.state.settings = cfg
        app.state.store = store
        app.state.registry = registry
        app.state.gate = gate
        app.state.runner = runner
        app.state.reconciled_at_startup = reconciled
        # Shared by every job-creating route (and the M6b-6 audition busy-check) so the
        # dedupe check-then-act cannot race a concurrent enqueue.
        app.state.enqueue_mutex = threading.Lock()
        # Serializes assignment writes against voice deletion (M6b-6): a voice must not
        # vanish between an assignment's validation and its durable write.
        app.state.voices_mutex = threading.Lock()
        try:
            yield
        finally:
            if not runner.stop(cancel_pending=True, timeout=_SHUTDOWN_TIMEOUT_SECONDS):
                # Logged, not raised (scoping doc): the daemon thread dies with the
                # process and the next startup's reconcile settles its row.
                logger.warning(
                    "job runner worker did not exit within %.0fs; its row settles at "
                    "the next startup reconcile",
                    _SHUTDOWN_TIMEOUT_SECONDS,
                )
            try:
                get_gpu_manager().free_all()
            except Exception:
                logger.exception("freeing the GPU at shutdown failed")

    app = FastAPI(title="Seiyuu API", version=__version__, lifespan=lifespan)
    register_error_handlers(app)
    app.include_router(system_routes.router, prefix="/api")
    app.include_router(engines_routes.router, prefix="/api")
    app.include_router(jobs_routes.router, prefix="/api")
    app.include_router(books_routes.router, prefix="/api")
    app.include_router(review_routes.router, prefix="/api")
    app.include_router(render_routes.router, prefix="/api")
    app.include_router(voices_routes.router, prefix="/api")
    return app


app = create_app()
