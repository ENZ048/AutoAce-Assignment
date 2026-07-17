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


def test_security_headers_on_every_response(client):
    r = client.post("/api/auth/login", json={"username": "x", "password": "y"})
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["referrer-policy"] == "no-referrer"
