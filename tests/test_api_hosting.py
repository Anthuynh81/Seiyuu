"""Public hosting: Cloudflare Access token check, Origin guard, frontend serving, `seiyuu serve`.

Tokens are signed here with a throwaway RSA key and checked by the real AccessVerifier, so the
signature, audience, issuer and expiry rules are exercised without reaching Cloudflare.
"""

import time

import jwt
import pytest
from click.testing import CliRunner
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
from starlette.applications import Starlette
from starlette.routing import WebSocketRoute
from starlette.websockets import WebSocketDisconnect

from seiyuu.api.access import AccessGuard, AccessVerifier, CertsUnavailable, build_verifier
from seiyuu.api.main import create_app
from seiyuu.api.origin import OriginGuard, origin_allowed
from seiyuu.cli import main
from seiyuu.settings import Settings
from test_api_m6b1 import make_settings

PUBLIC_HOST = "seiyuu.example.test"
ISSUER = "https://team.cloudflareaccess.com"
AUD = "aud-for-this-app"
KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
OTHER_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


class _SigningKey:
    key = KEY.public_key()


class _FakeKeys:
    def get_signing_key_from_jwt(self, token: str) -> _SigningKey:
        return _SigningKey()


def _token(key=KEY, **claims) -> str:
    payload = {
        "aud": AUD,
        "iss": ISSUER,
        "exp": int(time.time()) + 300,
        "email": "Owner@Example.test",
        **claims,
    }
    payload = {k: v for k, v in payload.items() if v is not None}
    return jwt.encode(payload, key, algorithm="RS256")


def _hosted_settings(tmp_path, **overrides) -> Settings:
    return make_settings(
        tmp_path,
        api_public_host=PUBLIC_HOST,
        api_access_team="team",
        api_access_aud=AUD,
        frontend_dist=tmp_path / "dist",
        **overrides,
    )


@pytest.fixture
def hosted(tmp_path):
    verifier = AccessVerifier(ISSUER, AUD, keys=_FakeKeys())
    app = create_app(settings=_hosted_settings(tmp_path), access_verifier=verifier)
    with TestClient(app, base_url=f"https://{PUBLIC_HOST}") as client:
        yield client


@pytest.fixture
def dist(tmp_path):
    root = tmp_path / "dist"
    (root / "assets").mkdir(parents=True)
    (root / "index.html").write_text("<!doctype html><title>Seiyuu UI</title>", encoding="utf-8")
    (root / "assets" / "app-abc123.js").write_text("console.log(1)", encoding="utf-8")
    (root / "favicon.svg").write_text("<svg/>", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("outside dist", encoding="utf-8")
    return root


def _signed(token: str | None = None) -> dict[str, str]:
    return {"Cf-Access-Jwt-Assertion": token or _token()}


# -- the token check ---------------------------------------------------------------------


def test_good_token_passes(hosted) -> None:
    resp = hosted.get("/api/health", headers=_signed())
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_missing_token_on_public_host_is_refused(hosted) -> None:
    resp = hosted.get("/api/health")
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "signed_out"
    assert resp.headers["cache-control"] == "no-store"


@pytest.mark.parametrize(
    "token",
    [
        pytest.param(_token(aud="another-apps-aud"), id="other-app-aud"),
        pytest.param(_token(iss="https://evil.cloudflareaccess.com"), id="other-team"),
        pytest.param(_token(exp=int(time.time()) - 60), id="expired"),
        pytest.param(_token(exp=None), id="no-expiry"),
        pytest.param(_token(email=None), id="service-token-no-email"),
        pytest.param(_token(key=OTHER_KEY), id="wrong-signature"),
        pytest.param("not-a-jwt", id="garbage"),
    ],
)
def test_bad_tokens_are_refused(hosted, token: str) -> None:
    resp = hosted.get("/api/health", headers=_signed(token))
    assert resp.status_code == 401


def test_unsigned_alg_none_token_is_refused(hosted) -> None:
    forged = jwt.encode({"aud": AUD, "iss": ISSUER, "email": "a@b.c"}, "", algorithm="none")
    assert hosted.get("/api/health", headers=_signed(forged)).status_code == 401


def test_verifier_returns_lowercase_email() -> None:
    verifier = AccessVerifier(ISSUER, AUD, keys=_FakeKeys())
    assert verifier.verify(_token()) == "owner@example.test"


def test_the_frontend_files_need_a_token_too(hosted, dist) -> None:
    assert hosted.get("/").status_code == 401
    assert hosted.get("/assets/app-abc123.js").status_code == 401
    assert hosted.get("/", headers=_signed()).status_code == 200


def test_rewritten_host_with_cloudflare_header_cannot_pass_as_local(tmp_path) -> None:
    verifier = AccessVerifier(ISSUER, AUD, keys=_FakeKeys())
    app = create_app(settings=_hosted_settings(tmp_path), access_verifier=verifier)
    with TestClient(app, base_url="http://localhost") as client:
        assert client.get("/api/health").status_code == 200
        for header in ("Cf-Ray", "Cf-Connecting-Ip"):
            assert client.get("/api/health", headers={header: "x"}).status_code == 401


def test_cloudflare_header_is_refused_even_when_hosting_is_not_configured(tmp_path) -> None:
    app = create_app(settings=make_settings(tmp_path))
    with TestClient(app, base_url="http://localhost") as client:
        assert client.get("/api/health", headers={"Cf-Ray": "x"}).status_code == 401


def test_public_host_is_not_answered_until_configured(tmp_path) -> None:
    app = create_app(settings=make_settings(tmp_path))
    with TestClient(app, base_url=f"https://{PUBLIC_HOST}") as client:
        assert client.get("/api/health").status_code == 400


def test_certs_outage_is_503_not_a_pass(tmp_path) -> None:
    class Down:
        def verify(self, token: str) -> str:
            raise CertsUnavailable("no network")

    app = create_app(settings=_hosted_settings(tmp_path), access_verifier=Down())
    with TestClient(app, base_url=f"https://{PUBLIC_HOST}") as client:
        resp = client.get("/api/health", headers=_signed())
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "sign_in_unavailable"


def test_api_description_stays_off_the_public_host(hosted) -> None:
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert hosted.get(path, headers=_signed()).status_code == 404


def test_api_description_still_works_locally(tmp_path) -> None:
    app = create_app(settings=make_settings(tmp_path))
    with TestClient(app) as client:
        assert client.get("/openapi.json").status_code == 200


def _websocket_app() -> Starlette:
    # Seiyuu has no WebSocket routes today; the guards must already cover any added later.
    async def echo(ws) -> None:
        await ws.accept()
        await ws.send_text(ws.url.path)
        await ws.close()

    return Starlette(routes=[WebSocketRoute("/ws", echo)])


def test_websocket_needs_a_token_too() -> None:
    verifier = AccessVerifier(ISSUER, AUD, keys=_FakeKeys())
    guarded = AccessGuard(_websocket_app(), public_host=PUBLIC_HOST, verifier=verifier)
    client = TestClient(guarded)
    public = {"Host": PUBLIC_HOST}  # websocket_connect ignores base_url's host
    with pytest.raises(WebSocketDisconnect) as refused:
        with client.websocket_connect("/ws", headers=public):
            pass
    assert refused.value.code == 1008
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws", headers={**public, **_signed("not-a-jwt")}):
            pass
    with client.websocket_connect("/ws", headers={**public, **_signed()}) as ws:
        assert ws.receive_text() == "/ws"


def test_websocket_from_another_site_is_closed() -> None:
    client = TestClient(OriginGuard(_websocket_app()))
    foreign = {"Host": PUBLIC_HOST, "Origin": "https://evil.example"}
    with pytest.raises(WebSocketDisconnect) as refused:
        with client.websocket_connect("/ws", headers=foreign):
            pass
    assert refused.value.code == 1008
    own = {"Host": PUBLIC_HOST, "Origin": f"https://{PUBLIC_HOST}"}
    with client.websocket_connect("/ws", headers=own) as ws:
        assert ws.receive_text() == "/ws"


# -- configuration ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("given", "named"),
    [
        ({"api_public_host": PUBLIC_HOST}, "API_ACCESS_TEAM, API_ACCESS_AUD"),
        ({"api_access_team": "team", "api_access_aud": AUD}, "API_PUBLIC_HOST"),
        ({"api_public_host": PUBLIC_HOST, "api_access_team": "team"}, "API_ACCESS_AUD"),
    ],
)
def test_partial_hosting_config_fails_loudly_naming_what_is_missing(tmp_path, given, named) -> None:
    with pytest.raises(ValueError, match=named):
        create_app(settings=make_settings(tmp_path, **given))


def test_no_hosting_config_means_no_verifier(tmp_path) -> None:
    assert build_verifier(make_settings(tmp_path)) is None


@pytest.mark.parametrize(
    "team",
    ["team", "TEAM ", "team.cloudflareaccess.com", "https://team.cloudflareaccess.com/"],
)
def test_team_may_be_written_short_or_in_full(tmp_path, team: str) -> None:
    assert make_settings(tmp_path, api_access_team=team).access_issuer == ISSUER


@pytest.mark.parametrize(
    "host",
    [PUBLIC_HOST, PUBLIC_HOST.upper(), f"https://{PUBLIC_HOST}/", f"{PUBLIC_HOST}:443"],
)
def test_public_host_is_normalized(tmp_path, host: str) -> None:
    assert make_settings(tmp_path, api_public_host=host).public_host == PUBLIC_HOST


def test_access_values_are_not_exposed_by_the_api(tmp_path) -> None:
    secret_aud = "aud-value-that-must-stay-private"
    verifier = AccessVerifier(ISSUER, secret_aud, keys=_FakeKeys())
    settings = make_settings(
        tmp_path, api_public_host=PUBLIC_HOST, api_access_team="team", api_access_aud=secret_aud
    )
    app = create_app(settings=settings, access_verifier=verifier)
    with TestClient(app, base_url="http://localhost") as client:
        for path in ("/api/settings", "/api/system"):
            assert secret_aud not in client.get(path).text


# -- the Origin guard ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("origin", "host", "allowed"),
    [
        (None, "localhost:8767", True),
        ("https://seiyuu.cyfa.dev", "seiyuu.cyfa.dev", True),
        ("https://seiyuu.cyfa.dev:443", "seiyuu.cyfa.dev", True),
        ("http://localhost:5173", "localhost:5173", True),
        ("http://127.0.0.1:8767", "127.0.0.1:8767", True),
        ("http://[::1]:8767", "[::1]:8767", True),
        ("https://evil.example", "seiyuu.cyfa.dev", False),
        ("https://hanatsu.cyfa.dev", "seiyuu.cyfa.dev", False),
        ("https://seiyuu.cyfa.dev.evil.example", "seiyuu.cyfa.dev", False),
        ("http://localhost:9999", "localhost:8767", False),
        ("null", "seiyuu.cyfa.dev", False),
        ("file://", "localhost", False),
        ("http://localhost:notaport", "localhost", False),
        ("http://localhost", None, False),
    ],
)
def test_origin_allowed(origin, host, allowed) -> None:
    assert origin_allowed(origin, host) is allowed


@pytest.fixture
def local(tmp_path):
    app = create_app(settings=make_settings(tmp_path, frontend_dist=tmp_path / "dist"))
    with TestClient(app, base_url="http://localhost:8767") as client:
        yield client


def test_foreign_origin_cannot_change_anything(local) -> None:
    resp = local.post(
        "/api/voices",
        json={"kind": "preset", "name": "x", "engine": "kokoro", "preset_id": "y"},
        headers={"Origin": "https://evil.example"},
    )
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == "bad_origin"
    assert local.get("/api/voices").json()["voices"] == []


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
def test_every_state_changing_method_is_guarded(local, method: str) -> None:
    resp = local.request(method, "/api/books/x", headers={"Origin": "https://evil.example"})
    assert resp.status_code == 403


def test_two_origin_headers_are_refused(local) -> None:
    resp = local.post(
        "/api/voices",
        headers=[("origin", "http://localhost:8767"), ("origin", "http://localhost:8767")],
    )
    assert resp.status_code == 403


def test_same_origin_post_still_works(local) -> None:
    resp = local.post(
        "/api/voices",
        json={"kind": "preset", "name": "x", "engine": "kokoro", "preset_id": "y"},
        headers={"Origin": "http://localhost:8767", "Sec-Fetch-Site": "same-origin"},
    )
    assert resp.status_code != 403


@pytest.mark.parametrize("site", ["cross-site", "same-site", "CROSS-SITE"])
def test_foreign_get_under_api_is_refused(local, site: str) -> None:
    resp = local.get("/api/engines/kokoro/preview", headers={"Sec-Fetch-Site": site})
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == "bad_origin"


@pytest.mark.parametrize("site", ["same-origin", "none"])
def test_own_and_typed_gets_pass(local, site: str) -> None:
    assert local.get("/api/health", headers={"Sec-Fetch-Site": site}).status_code == 200


def test_coming_back_from_the_login_page_can_load_the_app(local, dist) -> None:
    # Access redirects to / from its own domain, so this navigation is cross-site.
    assert local.get("/", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 200


# -- the frontend ---------------------------------------------------------------------------


def test_root_and_client_routes_serve_the_ui(local, dist) -> None:
    for path in ("/", "/voices", "/listen/"):
        resp = local.get(path)
        assert resp.status_code == 200
        assert "Seiyuu UI" in resp.text
        assert resp.headers["content-type"].startswith("text/html")
        assert resp.headers["cache-control"] == "no-cache"


def test_hashed_assets_are_immutable_with_a_correct_type(local, dist) -> None:
    resp = local.get("/assets/app-abc123.js")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/javascript")
    assert "immutable" in resp.headers["cache-control"]


def test_head_works(local, dist) -> None:
    assert local.head("/").status_code == 200


def test_missing_file_is_a_real_404_not_the_ui(local, dist) -> None:
    resp = local.get("/assets/gone.js")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


@pytest.mark.parametrize(
    "path",
    ["/..%2Fsecret.txt", "/assets/..%2F..%2Fsecret.txt", "/%2e%2e/secret.txt", "/..%5Csecret.txt"],
)
def test_traversal_cannot_leave_dist(local, dist, path: str) -> None:
    assert "outside dist" not in local.get(path).text


def test_unknown_api_path_keeps_the_envelope(local, dist) -> None:
    resp = local.get("/api/nonexistent")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


def test_wrong_method_keeps_its_405(local, dist) -> None:
    resp = local.delete("/api/health")
    assert resp.status_code == 405
    assert resp.json()["error"]["code"] == "method_not_allowed"
    assert "allow" in resp.headers


def test_post_to_a_ui_path_is_not_the_ui(local, dist) -> None:
    assert local.post("/voices").status_code == 404


def test_unbuilt_ui_says_how_to_build_it(local) -> None:
    resp = local.get("/")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "frontend_not_built"
    assert "npm" in resp.json()["error"]["message"]


# -- seiyuu serve ---------------------------------------------------------------------------


@pytest.fixture
def uvicorn_calls(monkeypatch, tmp_path):
    calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr("uvicorn.run", lambda *a, **kw: calls.append((a, kw)))
    settings = make_settings(tmp_path, frontend_dist=tmp_path / "dist")
    monkeypatch.setattr("seiyuu.settings.get_settings", lambda: settings)
    return calls


def test_serve_binds_loopback_with_one_worker(uvicorn_calls) -> None:
    result = CliRunner().invoke(main, ["serve"])
    assert result.exit_code == 0, result.output
    (args, kwargs) = uvicorn_calls[0]
    assert args == ("seiyuu.api.main:app",)
    assert kwargs == {"host": "127.0.0.1", "port": 8767, "workers": 1}
    assert "isn't built" in result.output


def test_serve_port_can_change_but_host_cannot(uvicorn_calls) -> None:
    assert CliRunner().invoke(main, ["serve", "--port", "9001"]).exit_code == 0
    assert uvicorn_calls[0][1]["port"] == 9001
    assert uvicorn_calls[0][1]["host"] == "127.0.0.1"
    assert CliRunner().invoke(main, ["serve", "--host", "0.0.0.0"]).exit_code != 0
