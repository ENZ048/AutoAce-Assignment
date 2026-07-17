import pytest

from dashboard.config import DashboardSettings, clear_settings_cache, get_dashboard_settings


def _set_required(monkeypatch):
    monkeypatch.setenv("DASHBOARD_ADMIN_USER", "autoace")
    monkeypatch.setenv("DASHBOARD_ADMIN_PASSWORD_HASH", "$2b$12$abcdefghijklmnopqrstuv")
    monkeypatch.setenv("DASHBOARD_JWT_SECRET", "s3cret")


def test_settings_load_from_env(monkeypatch, tmp_path):
    _set_required(monkeypatch)
    monkeypatch.setenv("DASHBOARD_DATA_DIR", str(tmp_path))
    clear_settings_cache()
    s = get_dashboard_settings()
    assert s.admin_user == "autoace"
    assert s.max_upload_mb == 1024  # default
    assert s.stub_analyze is False  # default
    assert str(s.data_dir) == str(tmp_path)


def test_missing_required_key_fails_fast(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DASHBOARD_ADMIN_USER", raising=False)
    monkeypatch.setenv("DASHBOARD_ADMIN_PASSWORD_HASH", "$2b$12$abcdefghijklmnopqrstuv")
    monkeypatch.setenv("DASHBOARD_JWT_SECRET", "s3cret")
    clear_settings_cache()
    with pytest.raises(Exception, match="(?i)admin_user"):
        get_dashboard_settings()


def test_settings_cached_until_cleared(monkeypatch, tmp_path):
    _set_required(monkeypatch)
    clear_settings_cache()
    first = get_dashboard_settings()
    monkeypatch.setenv("DASHBOARD_ADMIN_USER", "other")
    assert get_dashboard_settings() is first
    clear_settings_cache()
    assert get_dashboard_settings().admin_user == "other"


def test_hash_password_roundtrip():
    import bcrypt

    from dashboard.hash_password import make_hash

    h = make_hash("Trial#2026")
    assert h.startswith("$2b$")
    assert bcrypt.checkpw(b"Trial#2026", h.encode())
