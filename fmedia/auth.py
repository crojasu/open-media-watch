"""
auth.py — simple HTTP Basic Auth for the review panel.
No database needed: credentials live in .env as:
  REVIEW_USERS=alice:password1,bob:password2

Hardening on top of plain Basic Auth (require_reviewer guards ~75 routes
across main.py and dossiers.py):

  - Per-IP sliding-window rate limit, checked BEFORE credentials are looked
    at, so an attacker cannot brute-force passwords by flooding requests
    (even ones with valid credentials count toward the limit). Same
    minimalist in-memory pattern as fmedia/search.py's `_check_rate_limit`
    (no new dependency — no slowapi, no redis).

  - Origin-header check on state-changing methods (POST/PATCH/PUT/DELETE).
    HTTP Basic Auth has no SameSite-cookie-style cross-origin defense — a
    browser will happily re-attach cached Basic Auth credentials to a
    request triggered by another site. When a browser sends an Origin
    header that doesn't match this site, the request is rejected. When no
    Origin header is present at all (curl, the pytest TestClient, and many
    legitimate non-browser/same-origin clients never send one) the request
    is allowed through — this check must never become a new outage.
"""
from __future__ import annotations
import base64
import os
import secrets
import time
from collections import defaultdict
from urllib.parse import urlsplit

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from dotenv import load_dotenv

load_dotenv()

security = HTTPBasic()

# Public base URL of the API, used for the same-origin check; set SITE_URL in production. (Not imported from main.py, to
# avoid a circular import — main.py imports `require_reviewer` from here.
SITE_URL = os.getenv("SITE_URL", "http://localhost:8000")

STATE_CHANGING_METHODS = {"POST", "PATCH", "PUT", "DELETE"}

# ── Rate limiting: simple in-memory sliding window per IP ─────────────────────
# Same window as fmedia/search.py's /api/search limiter, reused for
# consistency. The cap is generous enough that normal review-panel usage
# (Basic Auth re-sends credentials on every request) never trips it, while
# still shutting down a credential-guessing flood.
RATE_LIMIT_WINDOW_S = 60
RATE_LIMIT_MAX_ATTEMPTS = 60
_rate_buckets: dict = defaultdict(list)


def _check_rate_limit(ip: str) -> bool:
    now = time.time()
    bucket = _rate_buckets[ip]
    cutoff = now - RATE_LIMIT_WINDOW_S
    while bucket and bucket[0] < cutoff:
        bucket.pop(0)
    if len(bucket) >= RATE_LIMIT_MAX_ATTEMPTS:
        return False
    bucket.append(now)
    return True


def reset_auth_rate_limits() -> None:
    """Test-only helper — clears all auth rate-limit buckets."""
    _rate_buckets.clear()


def _same_origin(origin_header: str, request: Request) -> bool:
    """True if `origin_header` (a browser-sent Origin value) matches this
    site — either the current request's own host, or the configured
    SITE_URL's host. Hostname-only comparison (ignoring scheme/port) so it
    stays correct behind a reverse proxy that may not preserve scheme."""
    origin_host = urlsplit(origin_header).hostname
    if not origin_host:
        return False
    site_host = urlsplit(SITE_URL).hostname
    return origin_host == request.url.hostname or origin_host == site_host


# Parse REVIEW_USERS env var: "alice:pass1,bob:pass2"
def _load_users() -> dict:
    raw = os.getenv("REVIEW_USERS")
    if not raw:
        return {}  # startup validation in main.py will catch this before any request is served
    users = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if ":" in pair:
            u, p = pair.split(":", 1)
            users[u.strip()] = p.strip()
    return users

USERS = _load_users()

# Constant-effort fallback so an unknown username takes the same code path
# (and roughly the same time) as a known username with a wrong password —
# otherwise `not stored_pw` short-circuits before compare_digest ever runs,
# letting a timing difference reveal which usernames exist.
_DUMMY_PASSWORD = secrets.token_urlsafe(32)


def require_reviewer(
    request: Request,
    credentials: HTTPBasicCredentials = Depends(security),
) -> str:
    """FastAPI dependency — returns username if valid, raises 401/403/429 otherwise."""
    client_ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many attempts — try again in a minute.",
        )

    if request.method in STATE_CHANGING_METHODS:
        origin = request.headers.get("origin")
        if origin and not _same_origin(origin, request):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cross-origin request rejected",
            )

    stored_pw = USERS.get(credentials.username)
    valid = secrets.compare_digest(
        credentials.password.encode(), (stored_pw or _DUMMY_PASSWORD).encode()
    )
    if not stored_pw or not valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


def try_reviewer(request: Request) -> "str | None":
    """Return the reviewer username if the request carries VALID Basic
    credentials, else None. Never raises and never counts toward the auth
    rate limit: used by public pages that only want to know whether the
    visitor is already logged in (e.g. /archive -> /archive-full)."""
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "basic" or not token:
        return None
    try:
        username, _, password = base64.b64decode(token).decode("utf-8").partition(":")
    except (ValueError, UnicodeDecodeError):
        return None
    stored_pw = USERS.get(username)
    if not stored_pw:
        return None
    return username if secrets.compare_digest(password.encode(), stored_pw.encode()) else None
