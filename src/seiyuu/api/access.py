"""Cloudflare Access: every request that came through the tunnel proves who it is.

The public hostname is on the internet and this API can spend money and delete books, so its
login can't be taken on trust: Access signs a JWT for each request and passes it in
Cf-Access-Jwt-Assertion. AccessGuard checks the signature, audience, issuer and expiry on HTTP
and WebSocket alike, on every path (the frontend's files too). Any failure refuses the request.

A request counts as through the tunnel when its Host is the public hostname or it carries a
header only Cloudflare adds, so a rewritten Host can't pass as local. Requests from this PC
have neither and pass without a token; ``seiyuu serve`` binds 127.0.0.1 only.

Docs: https://developers.cloudflare.com/cloudflare-one/access-controls/applications/http-apps/authorization-cookie/validating-json/
"""

import logging
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any, Protocol
from urllib.parse import urlsplit

import jwt
from starlette.concurrency import run_in_threadpool
from starlette.websockets import WebSocketClose

from seiyuu.api.errors import envelope
from seiyuu.settings import Settings

logger = logging.getLogger(__name__)

TOKEN_HEADER = b"cf-access-jwt-assertion"
CLOUDFLARE_HEADERS = frozenset({TOKEN_HEADER, b"cf-ray", b"cf-connecting-ip"})
# How long fetched signing keys are trusted. A token with an unknown key id fetches them at
# once, so a rotation needs no wait; this bounds how long a withdrawn key keeps working.
CERTS_TTL_S = 600
CERTS_TIMEOUT_S = 5
CLOSE_POLICY = 1008
# FastAPI's generated API description stays off the public hostname.
LOCAL_ONLY_PATHS = frozenset({"/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json"})

Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]
App = Callable[[Scope, Receive, Send], Awaitable[None]]


class CertsUnavailable(Exception):
    """Cloudflare's signing keys couldn't be fetched, so nothing can be checked right now."""


class InvalidAccessToken(Exception):
    """The token is missing a valid signature, audience, issuer, expiry or email."""


class Verifier(Protocol):
    def verify(self, token: str) -> str: ...


class AccessVerifier:
    """Checks one Access JWT and returns its email, lowercase."""

    def __init__(self, issuer: str, audience: str, keys: jwt.PyJWKClient | None = None) -> None:
        self.issuer = issuer
        self.audience = audience
        self.keys = keys or jwt.PyJWKClient(
            f"{issuer}/cdn-cgi/access/certs",
            cache_keys=False,
            lifespan=CERTS_TTL_S,
            timeout=CERTS_TIMEOUT_S,
        )

    def verify(self, token: str) -> str:
        try:
            key = self.keys.get_signing_key_from_jwt(token).key
            claims = jwt.decode(
                token,
                key,
                algorithms=["RS256"],
                audience=self.audience,
                issuer=self.issuer,
                options={"require": ["exp", "aud", "iss"]},
            )
        except jwt.PyJWKClientConnectionError as e:
            raise CertsUnavailable(str(e)) from e
        except jwt.PyJWTError as e:
            raise InvalidAccessToken(type(e).__name__) from e
        email = claims.get("email")
        if not isinstance(email, str) or "@" not in email:
            raise InvalidAccessToken("no email")  # a service token, not a person
        return email.strip().lower()


def build_verifier(settings: Settings) -> AccessVerifier | None:
    """None when public hosting isn't configured; half a configuration is a startup error, so
    a typo in .env can't look like a working login."""
    public_host = settings.public_host
    issuer = settings.access_issuer
    audience = settings.api_access_aud.strip()
    if not (public_host or issuer or audience):
        return None
    missing = [
        name
        for name, value in (
            ("API_PUBLIC_HOST", public_host),
            ("API_ACCESS_TEAM", issuer),
            ("API_ACCESS_AUD", audience),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            f"public hosting is only partly configured: set {', '.join(missing)} in .env, "
            "or unset all three API_PUBLIC_HOST / API_ACCESS_TEAM / API_ACCESS_AUD"
        )
    return AccessVerifier(issuer, audience)


def _host_name(host: str | None) -> str:
    try:
        return (urlsplit(f"//{host or ''}").hostname or "").lower()
    except ValueError:
        return ""


class AccessGuard:
    """Refuses a request from the tunnel without a good token (401, or 503 with no keys)."""

    def __init__(self, app: App, public_host: str, verifier: Verifier | None) -> None:
        self.app = app
        self.public_host = public_host
        self.verifier = verifier

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        host = token = None
        through_tunnel = False
        for k, v in scope.get("headers") or []:
            if k == b"host":
                host = v.decode("latin-1")
            elif k in CLOUDFLARE_HEADERS:
                through_tunnel = True
                if k == TOKEN_HEADER:
                    token = v.decode("latin-1")
        if self.public_host and _host_name(host) == self.public_host:
            through_tunnel = True
        if not through_tunnel:
            await self.app(scope, receive, send)
            return

        status = 401
        email: str | None = None
        if token and self.verifier is not None:
            try:
                email = await run_in_threadpool(self.verifier.verify, token)
            except InvalidAccessToken as e:
                logger.info("refused an Access token: %s", e)
            except CertsUnavailable as e:
                logger.warning("Access signing keys unavailable: %s", type(e.__cause__).__name__)
                status = 503
        if email is None:
            await self._refuse(scope, receive, send, status)
            return
        if scope.get("path") in LOCAL_ONLY_PATHS:
            response = envelope(404, "not_found", "Not Found")
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)

    @staticmethod
    async def _refuse(scope: Scope, receive: Receive, send: Send, status: int) -> None:
        if scope["type"] == "websocket":
            await WebSocketClose(code=CLOSE_POLICY)(scope, receive, send)
            return
        code, message = (
            ("signed_out", "Sign in again to use Seiyuu.")
            if status == 401
            else ("sign_in_unavailable", "Can't check your sign-in right now. Try again soon.")
        )
        response = envelope(status, code, message, headers={"Cache-Control": "no-store"})
        await response(scope, receive, send)
