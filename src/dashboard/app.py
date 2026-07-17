"""FastAPI app factory + background dispatcher. Static SPA serving arrives in Task 12."""

import asyncio
import logging
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI

from dashboard import api, runner, store
from dashboard.config import get_dashboard_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    settings = get_dashboard_settings()
    runner.sweep_orphans(app.state.db)
    stop = asyncio.Event()

    async def _dispatcher():
        db_path = settings.data_dir / "dashboard.db"
        while not stop.is_set():
            try:
                runner.dispatch_once(
                    app.state.db, db_path, app.state.jobs_dir, settings.stub_analyze
                )
            except Exception:  # noqa: BLE001 — the dispatcher must never die
                logger.exception("dispatcher tick failed")
            with suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=1.0)

    task = asyncio.create_task(_dispatcher())
    yield
    stop.set()
    await task


def create_app() -> FastAPI:
    settings = get_dashboard_settings()
    app = FastAPI(
        title="AutoAce Evaluation Dashboard",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=_lifespan,
    )
    app.state.db = store.connect(settings.data_dir / "dashboard.db")
    app.state.jobs_dir = settings.data_dir / "jobs"
    app.include_router(api.router)
    return app
