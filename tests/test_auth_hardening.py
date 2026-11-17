"""Tests for the require_reviewer hardening in fmedia/auth.py:
  - per-IP rate limiting on Basic Auth attempts (anti-brute-force)
  - Origin-header check on state-changing requests (anti-CSRF)
"""
import pytest

from fmedia.auth import RATE_LIMIT_MAX_ATTEMPTS, reset_auth_rate_limits

AUTH = ("tester", "testpass")


@pytest.fixture(autouse=True)
def _clean_auth_rate_limits():
    # Isolate this file's rate-limit assertions from the rest of the suite,
    # which shares the same in-memory per-IP bucket (TestClient always
    # reports the same client IP) — reset before AND after so later tests
    # in other files never inherit an exhausted bucket from this one.
    reset_auth_rate_limits()
    yield
    reset_auth_rate_limits()


_PAYLOAD = {"country": "XX", "city": "Test City", "incident_date": "2026-01-01", "is_test_case": True}


def test_rate_limit_blocks_rapid_wrong_password_attempts(client):
    """After RATE_LIMIT_MAX_ATTEMPTS rapid requests (even with wrong
    credentials), require_reviewer must start rejecting with 429 —
    before it even gets to checking the credentials."""
    seen_statuses = set()
    for _ in range(RATE_LIMIT_MAX_ATTEMPTS + 5):
        r = client.post("/api/dossiers", json=_PAYLOAD, auth=("tester", "wrong-password"))
        seen_statuses.add(r.status_code)

    assert 401 in seen_statuses  # the earlier wrong-password attempts
    assert 429 in seen_statuses  # the flood eventually gets throttled

    # The very next request — even with correct credentials — is still
    # throttled, proving this is an anti-brute-force limit, not just a
    # failed-login counter.
    r = client.post("/api/dossiers", json=_PAYLOAD, auth=AUTH)
    assert r.status_code == 429
    assert "Too many attempts" in r.json()["detail"]


def test_post_with_mismatched_origin_is_rejected(client):
    r = client.post(
        "/api/dossiers",
        json={"country": "cl", "city": "Ciudad Ficticia", "incident_date": "2026-03-14"},
        auth=AUTH,
        headers={"Origin": "https://evil-attacker.example"},
    )
    assert r.status_code == 403
    assert "Cross-origin" in r.json()["detail"]


def test_post_with_no_origin_header_still_works(client):
    # pytest's TestClient (like curl, and many non-browser clients) sends
    # no Origin header by default — this must keep working.
    r = client.post(
        "/api/dossiers",
        json={"country": "cl", "city": "Ciudad Ficticia", "incident_date": "2026-03-14",
              "is_test_case": True},
        auth=AUTH,
    )
    assert r.status_code == 201, r.text
