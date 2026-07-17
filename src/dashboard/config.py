"""Dashboard-only settings. Deliberately separate from autoace_audio.config."""

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DashboardSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DASHBOARD_", env_file=".env", extra="ignore")

    admin_user: str
    admin_password_hash: str
    jwt_secret: str
    max_upload_mb: int = 1024
    max_extract_mb: int = 4096  # decompressed-size budget for uploaded ZIPs (zip-bomb guard)
    stub_analyze: bool = False  # dev/test only: canned analyze, no models/keys
    data_dir: Path = Path("data")

    @field_validator("data_dir")
    @classmethod
    def _resolve_data_dir(cls, v: Path) -> Path:
        # Resolved once, at settings-load time, against the CWD then in effect —
        # otherwise a relative default silently relocates the DB/jobs dir/
        # batch_root.txt contents whenever the process's working directory changes.
        return v.resolve()


_cached: DashboardSettings | None = None


def get_dashboard_settings() -> DashboardSettings:
    global _cached
    if _cached is None:
        _cached = DashboardSettings()  # raises with the missing field named — fail fast
    return _cached


def clear_settings_cache() -> None:
    global _cached
    _cached = None
