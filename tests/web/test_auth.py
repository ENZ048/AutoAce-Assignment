import time

import jwt as pyjwt

from dashboard.auth import TOKEN_TTL_S, create_token, decode_token, verify_login
from dashboard.config import DashboardSettings
from dashboard.hash_password import make_hash


def make_settings(**over):
    base = dict(
        admin_user="autoace",
        admin_password_hash=make_hash("Right#Pass1"),
        jwt_secret="test-secret-0123456789abcdef-pad-to-32b",
    )
    base.update(over)
    return DashboardSettings(**base)


def test_verify_login_correct():
    assert verify_login("autoace", "Right#Pass1", make_settings()) is True


def test_verify_login_wrong_password_or_user():
    s = make_settings()
    assert verify_login("autoace", "wrong", s) is False
    assert verify_login("intruder", "Right#Pass1", s) is False


def test_malformed_stored_hash_is_auth_failure_not_error():
    s = make_settings(admin_password_hash="not-a-bcrypt-hash")
    assert verify_login("autoace", "Right#Pass1", s) is False


def test_token_roundtrip():
    s = make_settings()
    payload = decode_token(create_token(s), s)
    assert payload["sub"] == "autoace"
    assert payload["exp"] > time.time() + TOKEN_TTL_S - 120


def test_expired_token_rejected():
    s = make_settings()
    stale = pyjwt.encode(
        {"sub": "autoace", "exp": int(time.time()) - 10}, s.jwt_secret, algorithm="HS256"
    )
    assert decode_token(stale, s) is None


def test_garbage_and_wrong_secret_rejected():
    s = make_settings()
    assert decode_token("not.a.token", s) is None
    other = pyjwt.encode(
        {"sub": "autoace", "exp": int(time.time()) + 600},
        "other-secret-0123456789abcdef-pad-32b",
        algorithm="HS256",
    )
    assert decode_token(other, s) is None
