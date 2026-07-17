"""Generate the bcrypt hash for DASHBOARD_ADMIN_PASSWORD_HASH.

Usage: .venv/bin/python -m dashboard.hash_password '<plaintext>'
"""

import sys

import bcrypt


def make_hash(plaintext: str) -> str:
    return bcrypt.hashpw(plaintext.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m dashboard.hash_password '<plaintext>'")
    print(make_hash(sys.argv[1]))
