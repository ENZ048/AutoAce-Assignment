"""FastAPI app factory + background dispatcher, security headers, and single-origin SPA serving."""

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

    @app.middleware("http")
    async def _security_headers(request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    from pathlib import Path

    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    dist = Path(__file__).resolve().parents[2] / "webapp" / "dist"
    if dist.exists():  # dev API-only mode works without a build
        dist_real = dist.resolve()
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            # A literal ".." substring check isn't enough: a percent-encoded leading
            # slash (e.g. GET /%2Fetc%2Fpasswd -> path="/etc/passwd") makes `dist /
            # path` discard `dist` entirely, since pathlib resets the join on an
            # absolute right-hand side -- serving arbitrary local files. Resolve and
            # confirm containment instead, mirroring zipsafe.py's zip-slip guard.
            candidate = (dist / path).resolve()
            if path and candidate.is_relative_to(dist_real) and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(dist / "index.html")

    return app
