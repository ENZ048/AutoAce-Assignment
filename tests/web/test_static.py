from pathlib import Path

import pytest

DIST = Path(__file__).resolve().parents[2] / "webapp" / "dist"


@pytest.mark.skipif(not DIST.exists(), reason="webapp not built")
def test_spa_served_with_client_route_fallback(client):
    for path in ("/", "/jobs/deadbeef"):
        r = client.get(path)
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/html")


def test_api_404_stays_json_404(client, auth_header):
    r = client.get("/api/jobs/nope", headers=auth_header)
    assert r.status_code == 404
    assert r.json()["detail"] == "Job not found"


@pytest.mark.skipif(not DIST.exists(), reason="webapp not built")
def test_undefined_api_path_is_json_404_not_spa(client):
    r = client.get("/api/definitely/not/a/route")
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("application/json")


def test_security_headers_on_every_response(client):
    r = client.post("/api/auth/login", json={"username": "x", "password": "y"})
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["referrer-policy"] == "no-referrer"


def test_csp_header_present_and_strict(client, auth_header):
    r = client.get("/api/jobs", headers=auth_header)
    csp = r.headers["content-security-policy"]
    assert "default-src 'self'" in csp
    assert "object-src 'none'" in csp
    assert "frame-ancestors 'none'" in csp
    assert "script-src 'self'" in csp
    # no external hosts anywhere in the policy
    assert "http://" not in csp and "https://" not in csp
