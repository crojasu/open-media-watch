"""Unit tests: dossier data model + validation constants."""
from datetime import date

import pytest

from fmedia.database import (
    Dossier, DossierEntry, DossierFact, make_dossier_slug,
    JUDICIAL_STATUSES, DOSSIER_STATUSES, FACT_STATUSES, SHORT_QUOTE_MAX,
)
from tests.conftest import make_dossier


def test_slug_is_date_plus_city_never_name():
    slug = make_dossier_slug(date(2026, 3, 14), "Ciudad Ficticia")
    assert slug == "2026-03-ciudad-ficticia"


def test_slug_strips_accents_and_symbols():
    assert make_dossier_slug(date(2026, 1, 2), "Valparaíso (Región V)") == "2026-01-valparaiso-region-v"


def test_invalid_judicial_status_rejected():
    with pytest.raises(ValueError):
        Dossier(slug="x", country="CL", city="Ciudad Ficticia",
                incident_date=date(2026, 1, 1), judicial_status="condenado")


def test_invalid_dossier_status_rejected():
    with pytest.raises(ValueError):
        Dossier(slug="x", country="CL", city="Ciudad Ficticia",
                incident_date=date(2026, 1, 1), dossier_status="archivado")


def test_invalid_fact_status_rejected():
    with pytest.raises(ValueError):
        DossierFact(dossier_id=1, fact_text="dato", fact_status="rumor")


def test_fact_requires_valid_epistemic_status():
    for status in FACT_STATUSES:
        DossierFact(dossier_id=1, fact_text="dato", fact_status=status)  # no raise


def test_invalid_coverage_focus_rejected():
    with pytest.raises(ValueError):
        DossierEntry(dossier_id=1, entry_date=date(2026, 1, 1), outlet="X",
                     headline_paraphrase="t", coverage_focus="sensacionalista")


def test_short_quote_over_limit_rejected():
    with pytest.raises(ValueError):
        DossierEntry(dossier_id=1, entry_date=date(2026, 1, 1), outlet="X",
                     headline_paraphrase="t", short_quote="a" * (SHORT_QUOTE_MAX + 1))


def test_display_name_gated_by_public_flag(db):
    d = make_dossier(db, victim_name="Persona de Prueba", victim_name_public=False)
    assert d.display_name is None
    d.victim_name_public = True
    assert d.display_name == "Persona de Prueba"


def test_display_image_gated_by_rights_flag(db):
    d = make_dossier(db, image_url="https://example.org/x.jpg", image_rights_verified=False)
    assert d.display_image is None
    d.image_rights_verified = True
    assert d.display_image == "https://example.org/x.jpg"


def test_slug_unique_constraint(db):
    make_dossier(db)
    import sqlalchemy.exc
    with pytest.raises(sqlalchemy.exc.IntegrityError):
        make_dossier(db)  # same date+city → same slug


def test_status_vocabularies_are_closed_sets():
    assert JUDICIAL_STATUSES == {"investigacion", "juicio_en_curso", "sentencia_firme"}
    assert DOSSIER_STATUSES == {"en_seguimiento", "cobertura_cerrada", "en_espera_de_sentencia"}
    assert FACT_STATUSES == {"hecho_judicial", "alegacion_fiscalia", "version_medio_sin_confirmar"}
