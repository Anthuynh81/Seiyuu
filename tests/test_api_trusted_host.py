"""TrustedHost allowlist: the DNS-rebinding guard on the local API.

A malicious page can resolve its own domain to 127.0.0.1 and gain same-origin access
to the money-spending API; the Host allowlist (``api_allowed_hosts``) rejects such
requests before routing. The rejection is TrustedHostMiddleware's plain-text 400, NOT
the app's JSON envelope — it fires outside the router, and no legitimate client ever
sees it because the frontend's Host is always allowed.
"""

import pytest
from fastapi.testclient import TestClient

from seiyuu.api.main import create_app
from test_api_m6b1 import make_settings


@pytest.fixture
def client(tmp_path):
    settings = make_settings(tmp_path)
    app = create_app(settings=settings)
    with TestClient(app) as c:
        yield c


def test_rebound_host_rejected_on_read_route(client) -> None:
    resp = client.get("/api/system", headers={"Host": "evil.example.com"})
    assert resp.status_code == 400
    assert resp.text == "Invalid host header"  # middleware-level, pre-envelope


def test_rebound_host_rejected_on_paid_route(client) -> None:
    # Rejected before routing: no book lookup, no job enqueue, no paid-path reachable.
    resp = client.post(
        "/api/books/bk-x/attribute",
        json={"confirm_paid": True},
        headers={"Host": "evil.example.com"},
    )
    assert resp.status_code == 400
    assert resp.text == "Invalid host header"

    resp = client.delete("/api/books/bk-x", headers={"Host": "evil.example.com"})
    assert resp.status_code == 400


def test_default_allowed_hosts_reach_the_app(client) -> None:
    # No Host override: TestClient sends "testserver", which the default allows.
    assert client.get("/api/health").status_code == 200
    # Matching strips ports, so the Vite dev proxy (Host: localhost:5173) and a direct
    # uvicorn hit (Host: 127.0.0.1:8000) both pass under the bare-name defaults.
    for host in ("localhost", "127.0.0.1", "localhost:5173", "127.0.0.1:8000"):
        resp = client.get("/api/health", headers={"Host": host})
        assert resp.status_code == 200, host


def test_custom_allowlist_admits_lan_host(tmp_path) -> None:
    settings = make_settings(
        tmp_path, api_allowed_hosts="localhost, 127.0.0.1, testserver, tablet.lan"
    )
    with TestClient(create_app(settings=settings)) as c:
        assert c.get("/api/health", headers={"Host": "tablet.lan"}).status_code == 200
        assert c.get("/api/health", headers={"Host": "tablet.lan:8000"}).status_code == 200
        # The allowlist is exact: unlisted hosts stay rejected even with a custom value.
        assert c.get("/api/health", headers={"Host": "evil.example.com"}).status_code == 400
