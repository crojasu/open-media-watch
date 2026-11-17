"""
Test fixtures. Environment MUST be configured before any fmedia import:
database.py binds its engine and auth.py loads REVIEW_USERS at import time.
"""
import os
import sys
import tempfile
from pathlib import Path

# Isolated throwaway SQLite DB for the whole test session — never a real database
_TMP_DB = os.path.join(tempfile.mkdtemp(prefix="omw-tests-"), "test.db")
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP_DB}"
os.environ["REVIEW_USERS"] = "tester:testpass"
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-real")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from fmedia.database import Base, engine, SessionLocal  # noqa: E402
from fmedia.main import app  # noqa: E402

AUTH = ("tester", "testpass")


@pytest.fixture(autouse=True)
def fresh_db():
    """Blank schema per test — create_all covers the full current schema."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield


@pytest.fixture()
def db():
    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()


@pytest.fixture()
def client():
    # No context manager on purpose: skips lifespan (env validation + scheduler)
    return TestClient(app)


def make_dossier(db, **overrides):
    """Fictitious dossier factory — test data only, clearly marked."""
    from datetime import date
    from fmedia.database import Dossier, make_dossier_slug

    defaults = dict(
        country="CL",
        city="Ciudad Ficticia",
        incident_date=date(2026, 3, 14),
        victim_name="Persona de Prueba",
        victim_name_public=False,
        judicial_status="investigacion",
        dossier_status="en_seguimiento",
        is_test_case=True,
        published=True,
    )
    defaults.update(overrides)
    if "slug" not in defaults:
        defaults["slug"] = make_dossier_slug(defaults["incident_date"], defaults["city"])
    d = Dossier(**defaults)
    db.add(d)
    db.commit()
    db.refresh(d)
    return d
