"""All /api routes."""

import time

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from dashboard import store
from dashboard.auth import create_token, require_auth, verify_login
from dashboard.config import get_dashboard_settings

router = APIRouter(prefix="/api")


class LoginBody(BaseModel):
    username: str
    password: str


@router.post("/auth/login")
def login(body: LoginBody):
    settings = get_dashboard_settings()
    if not verify_login(body.username, body.password, settings):
        time.sleep(0.5)  # basic brute-force friction on failures only
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"access_token": create_token(settings)}


@router.get("/jobs")
def list_jobs(request: Request, user: str = Depends(require_auth)):
    return store.list_jobs(request.app.state.db)
