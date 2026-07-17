"""FastAPI app factory. Static SPA serving is added in Task 12."""

from fastapi import FastAPI

from dashboard import api, store
from dashboard.config import get_dashboard_settings


def create_app() -> FastAPI:
    settings = get_dashboard_settings()
    app = FastAPI(
        title="AutoAce Evaluation Dashboard",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,  # no public API browser
    )
    app.state.db = store.connect(settings.data_dir / "dashboard.db")
    app.state.jobs_dir = settings.data_dir / "jobs"
    app.include_router(api.router)
    return app
