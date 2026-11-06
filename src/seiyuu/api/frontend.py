"""Serve the built frontend from the API's own port, so one tunnel route reaches the whole app.

Done as the 404 path rather than a catch-all route, so an unknown /api path keeps its JSON
envelope and a wrong method keeps its 405. The dist folder is checked per request, so
rebuilding the frontend needs no backend restart.
"""

import mimetypes
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from seiyuu.api.errors import envelope, http_exception_envelope

# Explicit types for what Vite emits. Python's mimetypes reads the Windows registry, which can
# map .js to text/plain and break module scripts, so don't rely on it.
MEDIA_TYPES: dict[str, str] = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".map": "application/json",
    ".webmanifest": "application/manifest+json",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".webp": "image/webp",
    ".woff2": "font/woff2",
    ".wasm": "application/wasm",
    ".txt": "text/plain; charset=utf-8",
}
NO_CACHE = "no-cache"
IMMUTABLE = "public, max-age=31536000, immutable"  # Vite puts content hashes in these names
BUILD_COMMAND = "npm --prefix frontend run build"


def _media_type(path: Path) -> str:
    known = MEDIA_TYPES.get(path.suffix.lower())
    if known:
        return known
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


def _file_response(dist: Path, file: Path) -> FileResponse:
    rel = file.relative_to(dist).as_posix()
    cache = IMMUTABLE if rel.startswith("assets/") else NO_CACHE
    return FileResponse(file, media_type=_media_type(file), headers={"Cache-Control": cache})


def _resolve_static(dist: Path, url_path: str) -> Path | None:
    """Map a URL path to a file inside dist, refusing anything that escapes it."""
    rel = url_path.lstrip("/")
    if not rel:
        return None
    # Check the string before touching the filesystem. On Windows a backslash or drive colon
    # can turn dist / rel into an absolute path such as the UNC path //host/share, and
    # resolve() on that would open an outbound SMB connection before any confinement check.
    if "\\" in rel or ":" in rel or "\x00" in rel:
        return None
    parts = rel.split("/")
    if any(p in ("", ".", "..") for p in parts):
        return None
    try:
        candidate = dist.joinpath(*parts).resolve()
    except (OSError, ValueError):
        return None
    if not candidate.is_relative_to(dist) or not candidate.is_file():
        return None
    return candidate


def _is_api(path: str) -> bool:
    return path == "/api" or path.startswith("/api/")


def install_frontend(app: FastAPI, dist_dir: Path) -> None:
    """Answer a GET that matched no route with the built frontend. Replaces the app-wide
    HTTPException handler, whose envelope every other case still gets."""

    @app.exception_handler(StarletteHTTPException)
    async def _frontend_or_error(request: Request, exc: StarletteHTTPException) -> Response:
        path = request.url.path
        if exc.status_code != 404 or request.method not in ("GET", "HEAD") or _is_api(path):
            return http_exception_envelope(exc)

        dist = dist_dir.resolve()
        index = dist / "index.html"
        if not index.is_file():
            return envelope(
                404, "frontend_not_built", f"The UI isn't built yet. Run: {BUILD_COMMAND}"
            )
        file = _resolve_static(dist, path)
        if file is not None:
            return _file_response(dist, file)
        # A missing file (the last segment has an extension) is a real 404: serving index.html
        # for a missing .js gives confusing MIME errors in the browser.
        if "." in path.rstrip("/").rsplit("/", 1)[-1]:
            return http_exception_envelope(exc)
        return _file_response(dist, index)  # client-side routes such as /voices
