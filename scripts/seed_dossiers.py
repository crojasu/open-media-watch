"""
seed_dossiers.py — Seed FICTITIOUS case dossiers for local development.

GUARDRAIL: this script only creates clearly-marked test cases
(is_test_case=True, "Ciudad Ficticia"). It NEVER touches real cases
from the pipeline, and real case data must not be added here until
the feature is editorially validated.

Usage:
    venv/bin/python scripts/seed_dossiers.py
"""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fmedia.database import (  # noqa: E402
    init_db, SessionLocal, Dossier, DossierEntry, DossierFact, make_dossier_slug,
)


def seed():
    init_db()
    db = SessionLocal()
    try:
        if db.query(Dossier).filter(Dossier.is_test_case.is_(True)).count():
            print("Test dossiers already seeded — nothing to do.")
            return

        d1 = Dossier(
            slug=make_dossier_slug(date(2026, 3, 14), "Ciudad Ficticia"),
            country="CL",
            city="Ciudad Ficticia",
            region="Región de Prueba",
            incident_date=date(2026, 3, 14),
            victim_name="Persona de Prueba",
            victim_name_public=True,   # fictitious name, safe to display
            victim_age=34,
            judicial_status="investigacion",
            dossier_status="en_seguimiento",
            framing_analysis=(
                "TEST CASE — fictitious content. Early coverage avoided the legal "
                "term and used passive constructions; later coverage adopted "
                "the correct legal term and centered the victim, in line with the "
                "project style guide."
            ),
            # Image present but rights NOT verified → must never render:
            image_url="https://example.org/placeholder-test-image.jpg",
            image_rights_verified=False,
            is_test_case=True,
            published=True,
        )
        db.add(d1)
        db.flush()

        db.add_all([
            DossierEntry(
                dossier_id=d1.id, entry_date=date(2026, 3, 15),
                outlet="Diario Ficticio",
                headline_paraphrase=(
                    "Initial report describes the death of a woman in Ciudad Ficticia "
                    "without naming it relevant incident (test entry)."
                ),
                source_url="https://example.org/nota-1",
                uses_topic_term=False, uses_euphemism=True, coverage_focus="agresor",
            ),
            DossierEntry(
                dossier_id=d1.id, entry_date=date(2026, 3, 22),
                outlet="Portal de Prueba",
                headline_paraphrase=(
                    "Follow-up names the case as an alleged relevant incident and reports that "
                    "prosecutors opened an investigation (test entry)."
                ),
                source_url="https://example.org/nota-2",
                uses_topic_term=True, uses_euphemism=False, coverage_focus="victima",
                short_quote="La fiscalía investiga el caso.",
                quote_attribution="Portal de Prueba, 2026-03-22 (contenido ficticio)",
            ),
        ])

        db.add_all([
            DossierFact(
                dossier_id=d1.id,
                fact_text="A criminal investigation was opened on 2026-03-16 (test fact).",
                fact_status="hecho_judicial",
            ),
            DossierFact(
                dossier_id=d1.id,
                fact_text="Prosecutors allege the case qualifies as relevant incident (test fact).",
                fact_status="alegacion_fiscalia",
            ),
            DossierFact(
                dossier_id=d1.id,
                fact_text="One outlet reported an unconfirmed account of the events (test fact).",
                fact_status="version_medio_sin_confirmar",
            ),
        ])

        # Second dossier: unpublished — verifies public 404 behavior.
        d2 = Dossier(
            slug=make_dossier_slug(date(2026, 4, 2), "Villa Imaginaria"),
            country="AR",
            city="Villa Imaginaria",
            incident_date=date(2026, 4, 2),
            victim_name="Nombre Reservado",
            victim_name_public=False,  # name withheld → never rendered
            judicial_status="juicio_en_curso",
            dossier_status="en_espera_de_sentencia",
            is_test_case=True,
            published=False,
        )
        db.add(d2)
        db.commit()
        print(f"Seeded 2 fictitious dossiers: {d1.slug} (published), {d2.slug} (draft).")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
