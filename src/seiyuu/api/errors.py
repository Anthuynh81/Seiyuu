"""Uniform error envelope and typed-exception mapping (scoping doc section 1).

Every non-2xx body is ``{"error": {"code": str, "message": str, "detail": object|null}}``.
``message`` is the underlying typed exception's text verbatim — service messages are
user-facing and actionable by contract. ``code`` is a stable machine string so the M6c
client can branch without parsing prose. Route-specific refusals raise :class:`ApiError`
with their granular code; this module owns the app-wide handlers for the typed
exceptions that mean the same thing everywhere (JobNotFoundError -> 404,
IllegalTransitionError -> 409, lock/DB contention -> 503).
"""

import logging
import sqlite3

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from seiyuu.repository import IllegalTransitionError, JobNotFoundError, RepositoryError

logger = logging.getLogger(__name__)


class ApiError(Exception):
    """A route-level refusal with an explicit status and stable machine code."""

    def __init__(self, status: int, code: str, message: str, detail: object = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.detail = detail


def envelope(
    status: int, code: str, message: str, detail: object = None, headers: dict | None = None
) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message, "detail": jsonable_encoder(detail)}},
        headers=headers,
    )


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        return envelope(exc.status, exc.code, exc.message, exc.detail)

    @app.exception_handler(JobNotFoundError)
    async def _job_not_found(request: Request, exc: JobNotFoundError) -> JSONResponse:
        return envelope(404, "not_found", str(exc))

    @app.exception_handler(IllegalTransitionError)
    async def _illegal_transition(request: Request, exc: IllegalTransitionError) -> JSONResponse:
        detail = {"job_id": exc.job_id, "current": exc.current.value, "target": exc.target.value}
        return envelope(409, "illegal_transition", str(exc), detail)

    @app.exception_handler(RepositoryError)
    async def _repository_error(request: Request, exc: RepositoryError) -> JSONResponse:
        # File-lock waiter timeouts and registry contention: retryable, not a client bug.
        return envelope(503, "lock_timeout", str(exc), headers={"Retry-After": "1"})

    @app.exception_handler(sqlite3.OperationalError)
    async def _sqlite_error(request: Request, exc: sqlite3.OperationalError) -> JSONResponse:
        if "locked" in str(exc).lower() or "busy" in str(exc).lower():
            return envelope(503, "lock_timeout", str(exc), headers={"Retry-After": "1"})
        return envelope(500, "internal", str(exc))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return envelope(422, "invalid", "request validation failed", exc.errors())

    @app.exception_handler(StarletteHTTPException)
    async def _http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        # Framework-raised 404 (unmatched path) / 405 (wrong method) must keep the
        # envelope contract too; headers pass through so 405's Allow survives.
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return envelope(exc.status_code, code, str(exc.detail), headers=exc.headers)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Catch-all: keep the envelope even for bugs. Generic message only — arbitrary
        # exception text can carry paths or key material, unlike the typed service
        # exceptions whose messages are user-facing by contract. The traceback still
        # reaches the server log (and uvicorn re-raises via ServerErrorMiddleware).
        logger.exception("unhandled error on %s %s", request.method, request.url.path)
        return envelope(500, "internal", f"unhandled {type(exc).__name__}")
