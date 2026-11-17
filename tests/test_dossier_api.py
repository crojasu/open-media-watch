"""API + integration tests for the dossier feature."""
from tests.conftest import make_dossier

AUTH = ("tester", "testpass")


def test_create_dossier_requires_auth(client):
    r = client.post("/api/dossiers", json={
        "country": "cl", "city": "Ciudad Ficticia", "incident_date": "2026-03-14",
    })
    assert r.status_code == 401


def test_create_and_publish_flow(client):
    r = client.post("/api/dossiers", json={
        "country": "cl", "city": "Ciudad Ficticia", "incident_date": "2026-03-14",
        "is_test_case": True,
    }, auth=AUTH)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["slug"] == "2026-03-ciudad-ficticia"
    assert body["country"] == "CL"          # normalized to uppercase
    assert body["published"] is False       # draft by default

    # Draft not public yet
    assert client.get(f"/api/dossiers/by-slug/{body['slug']}").status_code == 404

    # Publish → public
    r2 = client.post(f"/api/dossiers/{body['id']}/publish", auth=AUTH)
    assert r2.status_code == 200 and r2.json()["published"] is True
    assert client.get(f"/api/dossiers/by-slug/{body['slug']}").status_code == 200


def test_slug_collision_gets_suffix(client):
    payload = {"country": "CL", "city": "Ciudad Ficticia",
               "incident_date": "2026-03-14", "is_test_case": True}
    s1 = client.post("/api/dossiers", json=payload, auth=AUTH).json()["slug"]
    s2 = client.post("/api/dossiers", json=payload, auth=AUTH).json()["slug"]
    assert s1 == "2026-03-ciudad-ficticia"
    assert s2 == "2026-03-ciudad-ficticia-2"


def test_patch_judicial_status(client, db):
    d = make_dossier(db)
    r = client.patch(f"/api/dossiers/{d.id}",
                     json={"judicial_status": "juicio_en_curso"}, auth=AUTH)
    assert r.status_code == 200
    assert r.json()["judicial_status"] == "juicio_en_curso"

    bad = client.patch(f"/api/dossiers/{d.id}",
                       json={"judicial_status": "linchamiento"}, auth=AUTH)
    assert bad.status_code == 422






