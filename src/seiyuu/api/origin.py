"""Origin checks: a page on another site must not drive the API through the owner's browser.

TrustedHostMiddleware stops DNS rebinding (a foreign Host header) and AccessGuard checks the
login, but a browser attaches its Access cookie to a request another site makes it send, so
neither stops a foreign page. Browsers send Origin on every WebSocket handshake and every
POST/PUT/PATCH/DELETE, and that Origin must be this very host and port.

Several GET routes do real work too (auditions and previews synthesize, read-along runs
whisper) and a cross-site GET carries no Origin, so anything under /api also refuses a
Sec-Fetch-Site of "cross-site" or "same-site" (another cyfa.dev app, or another localhost port).
Non-browser clients send neither header and pass; through the tunnel they still need a token.
"""

from collections.abc import MutableMapping
from typing import Any
from urllib.parse import urlsplit

from starlette.websockets import WebSocketClose

from seiyuu.api.access import CLOSE_POLICY, App, Receive, Scope, Send
from seiyuu.api.errors import envelope

STATE_CHANGING = frozenset({"POST", "PUT", "PATCH", "DELETE"})
FOREIGN_FETCH_SITES = frozenset({"cross-site", "same-site"})
DEFAULT_PORTS = {"http": 80, "https": 443}


def origin_allowed(origin: str | None, host_header: str | None) -> bool:
    """No Origin (not a browser), or the same host and port as the Host header.

    The Vite dev proxy keeps the browser's Host, so localhost:5173 matches itself. A scheme
    without a host, including the opaque origin "null", is refused."""
    if origin is None:
        return True
    try:
        parts = urlsplit(origin.strip())
        default_port = DEFAULT_PORTS.get(parts.scheme)
        if default_port is None or not parts.hostname:
            return False
        served = urlsplit(f"//{host_header or ''}")
        return (parts.hostname.lower(), parts.port or default_port) == (
            (served.hostname or "").lower(),
            served.port or default_port,
        )
    except ValueError:  # a malformed port or bracket in either header
        return False


def _is_api(path: str) -> bool:
    return path == "/api" or path.startswith("/api/")


def _headers(scope: MutableMapping[str, Any]) -> tuple[str | None, str | None, str | None]:
    origin = host = fetch_site = None
    for k, v in scope.get("headers") or []:
        if k == b"origin":
            # Two Origin headers: not a browser we trust ("null" is always refused).
            origin = "null" if origin is not None else v.decode("latin-1")
        elif k == b"host":
            host = v.decode("latin-1")
        elif k == b"sec-fetch-site":
            fetch_site = v.decode("latin-1").strip().lower()
    return origin, host, fetch_site


class OriginGuard:
    """Closes a foreign WebSocket handshake; 403 for a foreign request under /api."""

    def __init__(self, app: App) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "websocket":
            origin, host, _ = _headers(scope)
            if not origin_allowed(origin, host):
                await WebSocketClose(code=CLOSE_POLICY)(scope, receive, send)
                return
        elif scope["type"] == "http" and _is_api(scope.get("path", "")):
            origin, host, fetch_site = _headers(scope)
            state_changing = scope.get("method", "").upper() in STATE_CHANGING
            if fetch_site in FOREIGN_FETCH_SITES or (
                state_changing and not origin_allowed(origin, host)
            ):
                response = envelope(
                    403,
                    "bad_origin",
                    "Requests from another site can't use Seiyuu. Open Seiyuu and try again.",
                    headers={"Cache-Control": "no-store"},
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)
