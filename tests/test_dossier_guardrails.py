"""
Ethical guardrails — BLOCKING acceptance criteria.
These tests exist so a regression on any guardrail fails CI, not review.
"""
import re
from datetime import date
from pathlib import Path

from fmedia.database import Dossier, DossierEntry, DossierFact
from tests.conftest import make_dossier

TEMPLATES = Path(__file__).resolve().parent.parent / "fmedia" / "templates"


# ── 1. Images require verified rights ─────────────────────────────────────────

def test_image_not_rendered_without_rights_flag(client, db):
    d = make_dossier(db, image_url="https://example.org/secret-photo.jpg",
                     image_rights_verified=False)
    html = client.get(f"/dossier/{d.slug}").text
    assert "secret-photo.jpg" not in html




# ── 2. Victim name only when public ───────────────────────────────────────────





# ── 2b. Victim age / city / region only when explicitly public ───────────────
# Age + city + region together can re-identify a victim even without a name.

def test_victim_age_and_location_not_rendered_by_default(client, db):
    d = make_dossier(db, victim_age=41, city="Guadalajara Test", region="Jalisco Test")
    assert d.display_age is None
    assert d.display_city is None
    assert d.display_region is None
    html = client.get(f"/dossier/{d.slug}").text
    assert "41" not in html
    assert "Guadalajara Test" not in html
    assert "Jalisco Test" not in html




# ── 3. No fields that could identify minors ───────────────────────────────────

MINOR_PATTERNS = re.compile(r"(child|minor|kid|hij[oa]|menor|niñ|infan)", re.IGNORECASE)

def test_dossier_tables_have_no_minor_identifying_columns():
    for model in (Dossier, DossierEntry, DossierFact):
        for col in model.__table__.columns:
            assert not MINOR_PATTERNS.search(col.name), (
                f"{model.__tablename__}.{col.name} podría identificar menores — prohibido"
            )


def test_dossier_model_has_no_perpetrator_name_field():
    cols = {c.name for c in Dossier.__table__.columns}
    assert not any("perpetrator" in c or "agresor" in c for c in cols), (
        "El expediente no debe almacenar identidad del acusado — presunción de inocencia"
    )


# ── 4. Fact/allegation separation is mandatory ────────────────────────────────

def test_fact_status_column_not_nullable():
    assert DossierFact.__table__.columns["fact_status"].nullable is False


def test_api_rejects_fact_without_valid_status(client, db):
    d = make_dossier(db)
    r = client.post(f"/api/dossiers/{d.id}/facts",
                    json={"fact_text": "algo pasó", "fact_status": "texto_libre"},
                    auth=("tester", "testpass"))
    assert r.status_code == 422


# ── 5. Quotes: short + attributed only ────────────────────────────────────────

def test_api_rejects_quote_without_attribution(client, db):
    d = make_dossier(db)
    r = client.post(f"/api/dossiers/{d.id}/entries",
                    json={"entry_date": "2026-03-15", "outlet": "Diario Ficticio",
                          "headline_paraphrase": "titular reformulado",
                          "short_quote": "cita sin atribución"},
                    auth=("tester", "testpass"))
    assert r.status_code == 422


# ── 6. Presumption of innocence in rendered copy ──────────────────────────────



def test_presumption_note_absent_after_final_sentence(client, db):
    d = make_dossier(db, judicial_status="sentencia_firme")
    html = client.get(f"/dossier/{d.slug}").text
    assert "Presumption of innocence" not in html


def test_api_responses_never_expose_victim_name(client, db):
    """La API pública de escritura devuelve metadatos, jamás el nombre."""
    r = client.post("/api/dossiers", json={
        "country": "CL", "city": "Ciudad Ficticia", "incident_date": "2026-03-14",
        "victim_name": "Nombre Que No Debe Salir", "victim_name_public": False,
        "is_test_case": True,
    }, auth=("tester", "testpass"))
    assert r.status_code == 201
    assert "Nombre Que No Debe Salir" not in r.text
    assert "victim_name" not in r.json()

    d_id = r.json()["id"]
    for resp in (
        client.patch(f"/api/dossiers/{d_id}", json={"region": "Región de Prueba"},
                     auth=("tester", "testpass")),
        client.post(f"/api/dossiers/{d_id}/publish", auth=("tester", "testpass")),
    ):
        assert "Nombre Que No Debe Salir" not in resp.text




def test_quote_attribution_enforced_at_schema_level():
    from fmedia.database import DossierEntry
    names = {c.name for c in DossierEntry.__table__.constraints if c.name}
    assert "ck_entry_quote_requires_attribution" in names




# ── 7. Visibility: unpublished dossiers are private ───────────────────────────

def test_unpublished_dossier_returns_404(client, db):
    d = make_dossier(db, published=False)
    assert client.get(f"/dossier/{d.slug}").status_code == 404




# ── 8. Test cases clearly marked ──────────────────────────────────────────────



def test_seed_script_only_creates_marked_test_cases():
    import ast
    seed_path = Path(__file__).resolve().parent.parent / "scripts" / "seed_dossiers.py"
    tree = ast.parse(seed_path.read_text(encoding="utf-8"))
    dossier_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", getattr(node.func, "attr", "")) == "Dossier"
    ]
    assert dossier_calls, "El seed debe crear al menos un dossier de prueba"
    for call in dossier_calls:
        kwargs = {k.arg: k.value for k in call.keywords}
        flag = kwargs.get("is_test_case")
        assert isinstance(flag, ast.Constant) and flag.value is True, (
            "Todo Dossier() del seed debe llevar is_test_case=True — nunca casos reales"
        )


# ── Public JSON API guardrails (tests rewritten for the API-only repository) ──

def _detail(client, d):
    r = client.get(f"/api/dossiers/by-slug/{d.slug}")
    assert r.status_code == 200, r.text
    return r.json()


def test_api_hides_victim_name_unless_public(client, db):
    d = make_dossier(db, victim_name="Nombre Reservado Test", victim_name_public=False)
    body = _detail(client, d)
    assert body["display_name"] is None
    assert "Nombre Reservado Test" not in client.get("/api/dossiers").text
    assert "Nombre Reservado Test" not in str(body)


def test_api_shows_victim_name_when_public(client, db):
    d = make_dossier(db, victim_name="Persona de Prueba", victim_name_public=True)
    assert _detail(client, d)["display_name"] == "Persona de Prueba"


def test_api_hides_age_and_location_unless_public(client, db):
    d = make_dossier(db, victim_age=34, victim_age_public=False, region="Region Test",
                     location_public=False)
    body = _detail(client, d)
    assert body["victim_age"] is None and body["city"] is None and body["region"] is None
    d2 = make_dossier(db, slug="2026-03-otra", victim_age=34, victim_age_public=True,
                      region="Region Test", location_public=True)
    body2 = _detail(client, d2)
    assert body2["victim_age"] == 34 and body2["city"] == "Ciudad Ficticia"


def test_api_image_only_with_verified_rights(client, db):
    d = make_dossier(db, image_url="https://example.org/x.jpg", image_rights_verified=False)
    assert _detail(client, d)["display_image"] is None
    d2 = make_dossier(db, slug="2026-03-img", image_url="https://example.org/x.jpg",
                      image_rights_verified=True)
    assert _detail(client, d2)["display_image"] == "https://example.org/x.jpg"


def test_api_lists_only_published(client, db):
    make_dossier(db, slug="2026-03-pub", published=True)
    make_dossier(db, slug="2026-03-draft", published=False)
    slugs = [x["slug"] for x in client.get("/api/dossiers").json()]
    assert slugs == ["2026-03-pub"]
    assert client.get("/api/dossiers/by-slug/2026-03-draft").status_code == 404


def test_api_never_exposes_raw_victim_fields(client, db):
    d = make_dossier(db, victim_name="Nombre Reservado Test", victim_name_public=False,
                     image_url="https://example.org/secret.jpg")
    keys = set(_detail(client, d))
    assert "victim_name" not in keys and "image_url" not in keys
