"""Bridge pipeline→dossier: create draft from Case + suggested timeline entries."""
from datetime import date, datetime

from fmedia.database import Case, Article, Dossier
from tests.conftest import make_dossier

AUTH = ("tester", "testpass")


def _make_case(db, **overrides):
    """Fictitious pipeline Case — test data only."""
    defaults = dict(
        victim_name="Persona de Prueba",
        victim_age=31,
        country="CL",
        incident_location="Ciudad Ficticia",
        incident_region="Región de Prueba",
        incident_date=date(2026, 2, 10),
        event_type="category_a",
        article_count=2,
        first_reported_at=datetime(2026, 2, 11, 8, 0),
    )
    defaults.update(overrides)
    c = Case(**defaults)
    db.add(c)
    db.commit()
    db.refresh(c)
    return c


def _make_article(db, case_id, i=1, **overrides):
    defaults = dict(
        guardian_id=f"test-art-{case_id}-{i}",
        title=f"Titular original de prueba {i}",
        web_url=f"https://example.org/case{case_id}/art{i}",
        source="Diario Ficticio",
        published_date=date(2026, 2, 10 + i),
        country="CL",
        review_status="confirmed",
        event_type="category_a",
        case_id=case_id,
    )
    defaults.update(overrides)
    a = Article(**defaults)
    db.add(a)
    db.commit()
    return a


def test_from_case_creates_prefilled_draft(client, db):
    c = _make_case(db)
    r = client.post(f"/api/dossiers/from-case/{c.id}", auth=AUTH)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["slug"] == "2026-02-ciudad-ficticia"
    assert body["country"] == "CL"
    assert body["published"] is False              # always a draft
    assert body["victim_name_public"] is False     # GUARDRAIL: never public by default
    assert "victim_name" not in body               # GUARDRAIL: API never leaks the name

    d = db.get(Dossier, body["id"])
    assert d.case_id == c.id
    assert d.victim_name == "Persona de Prueba"
    assert d.victim_age == 31


def test_from_case_never_copies_perpetrator_or_children(client, db):
    c = _make_case(db, perpetrator_name="Acusado Ficticio", perpetrator_age=40,
                   children_orphaned=2)
    r = client.post(f"/api/dossiers/from-case/{c.id}", auth=AUTH)
    assert r.status_code == 201
    d = db.get(Dossier, r.json()["id"])
    cols = {col.name for col in Dossier.__table__.columns}
    assert not any("perpetrator" in x or "children" in x for x in cols)
    assert "Acusado Ficticio" not in r.text


def test_from_case_duplicate_returns_409(client, db):
    c = _make_case(db)
    assert client.post(f"/api/dossiers/from-case/{c.id}", auth=AUTH).status_code == 201
    r = client.post(f"/api/dossiers/from-case/{c.id}", auth=AUTH)
    assert r.status_code == 409


def test_from_case_missing_case_404(client):
    assert client.post("/api/dossiers/from-case/99999", auth=AUTH).status_code == 404


def test_from_case_falls_back_to_first_reported_date(client, db):
    c = _make_case(db, incident_date=None)
    r = client.post(f"/api/dossiers/from-case/{c.id}", auth=AUTH)
    assert r.status_code == 201
    assert r.json()["slug"].startswith("2026-02-")


def test_suggested_entries_lists_linked_articles(client, db):
    c = _make_case(db)
    _make_article(db, c.id, i=1)
    _make_article(db, c.id, i=2)
    _make_article(db, c.id, i=3, review_status="pending")  # not confirmed → excluded

    d_id = client.post(f"/api/dossiers/from-case/{c.id}", auth=AUTH).json()["id"]
    r = client.get(f"/api/dossiers/{d_id}/suggested-entries", auth=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["pending"] == 2
    assert {s["outlet"] for s in body["suggestions"]} == {"Diario Ficticio"}
    assert all(s["original_title"].startswith("Titular original") for s in body["suggestions"])


def test_suggested_entries_excludes_already_added(client, db):
    c = _make_case(db)
    _make_article(db, c.id, i=1)
    _make_article(db, c.id, i=2)
    d_id = client.post(f"/api/dossiers/from-case/{c.id}", auth=AUTH).json()["id"]

    # Reviewer adds a paraphrased entry for article 1
    client.post(f"/api/dossiers/{d_id}/entries", json={
        "entry_date": "2026-02-11", "outlet": "Diario Ficticio",
        "headline_paraphrase": "Cobertura inicial del caso (reformulada, prueba)",
        "source_url": f"https://example.org/case{c.id}/art1",
    }, auth=AUTH)

    body = client.get(f"/api/dossiers/{d_id}/suggested-entries", auth=AUTH).json()
    assert body["pending"] == 1
    assert body["suggestions"][0]["source_url"].endswith("art2")


def test_suggested_entries_requires_auth(client, db):
    d = make_dossier(db)
    assert client.get(f"/api/dossiers/{d.id}/suggested-entries").status_code == 401


def test_original_titles_never_reach_public_page(client, db):
    """El título original es solo para el reviewer — jamás en la página pública."""
    c = _make_case(db)
    _make_article(db, c.id, i=1)
    d_id = client.post(f"/api/dossiers/from-case/{c.id}", auth=AUTH).json()["id"]
    client.post(f"/api/dossiers/{d_id}/publish", auth=AUTH)
    slug = db.get(Dossier, d_id).slug
    html = client.get(f"/dossier/{slug}").text
    assert "Titular original de prueba" not in html