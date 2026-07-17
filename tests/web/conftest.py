import zipfile

import pytest
from fastapi.testclient import TestClient

from dashboard.config import clear_settings_cache
from dashboard.hash_password import make_hash

PASSWORD = "Right#Pass1"
PASSWORD_HASH = make_hash(PASSWORD)  # once per session — bcrypt is deliberately slow


@pytest.fixture()
def app_env(monkeypatch, tmp_path):
    monkeypatch.setenv("DASHBOARD_ADMIN_USER", "autoace")
    monkeypatch.setenv("DASHBOARD_ADMIN_PASSWORD_HASH", PASSWORD_HASH)
    monkeypatch.setenv("DASHBOARD_JWT_SECRET", "test-secret-0123456789abcdef-pad-to-32b")
    monkeypatch.setenv("DASHBOARD_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DASHBOARD_STUB_ANALYZE", "1")
    clear_settings_cache()
    yield tmp_path
    clear_settings_cache()


@pytest.fixture()
def client(app_env):
    from dashboard.app import create_app

    with TestClient(create_app()) as c:
        yield c


@pytest.fixture()
def auth_header(client):
    r = client.post("/api/auth/login", json={"username": "autoace", "password": PASSWORD})
    assert r.status_code == 200
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def make_batch_zip(path, names=("call_001.wav", "call_002.wav"), manifest_rows=None, manifest=True):
    """Tiny fake batch: content is never decoded before processing, so b'RIFF' suffices."""
    with zipfile.ZipFile(path, "w") as zf:
        for n in names:
            zf.writestr(n, b"RIFF0000WAVEfake")
        if manifest:
            rows = manifest_rows if manifest_rows is not None else [f"{n}," for n in names]
            zf.writestr("labels.csv", "name,result_json\n" + "\n".join(rows) + "\n")
    return path
