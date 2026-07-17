"""Single-admin auth: bcrypt against the env-provisioned hash, stateless HS256 JWT."""

import time

import bcrypt
import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from dashboard.config import DashboardSettings, get_dashboard_settings

TOKEN_TTL_S = 24 * 3600

_bearer = HTTPBearer(auto_error=False)


def verify_login(username: str, password: str, settings: DashboardSettings) -> bool:
    if username != settings.admin_user:
        return False
    try:
        return bcrypt.checkpw(
            password.encode("utf-8"), settings.admin_password_hash.encode("utf-8")
        )
    except ValueError:  # malformed stored hash — treat as auth failure, never a 500
        return False


def create_token(settings: DashboardSettings) -> str:
    payload = {"sub": settings.admin_user, "exp": int(time.time()) + TOKEN_TTL_S}
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def decode_token(token: str, settings: DashboardSettings) -> dict | None:
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
    except jwt.PyJWTError:
        return None


def require_auth(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    if creds is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    payload = decode_token(creds.credentials, get_dashboard_settings())
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return payload["sub"]
