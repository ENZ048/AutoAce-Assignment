"""Dashboard-only settings. Deliberately separate from autoace_audio.config."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class DashboardSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DASHBOARD_", env_file=".env", extra="ignore")

    admin_user: str
    admin_password_hash: str
    jwt_secret: str
    max_upload_mb: int = 1024
    stub_analyze: bool = False  # dev/test only: canned analyze, no models/keys
    data_dir: Path = Path("data")


_cached: DashboardSettings | None = None


def get_dashboard_settings() -> DashboardSettings:
    global _cached
    if _cached is None:
        _cached = DashboardSettings()  # raises with the missing field named — fail fast
    return _cached


def clear_settings_cache() -> None:
    global _cached
    _cached = None
