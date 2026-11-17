"""Canonical source names at write time (fmedia/database.py)."""
from datetime import date

from fmedia.database import save_article, _canonical_source, Article


def _article(guardian_id, source):
    return {
        "guardian_id": guardian_id,
        "title": f"Test article {guardian_id}",
        "published_date": date(2026, 1, 1),
        "country": "AR",
        "source": source,
    }


def test_canonical_source_known_variants():
    assert _canonical_source("Clarin.com") == "Clarín"
    assert _canonical_source("Milenio") == "MILENIO"
    assert _canonical_source("Cooperativa.cl") == "Cooperativa"
    assert _canonical_source("Animal Politico") == "Animal Político"


def test_canonical_source_passthrough_and_none():
    assert _canonical_source("Infobae") == "Infobae"
    assert _canonical_source(None) is None
    assert _canonical_source("") == ""


def test_save_article_normalizes_source(db):
    row = save_article(db, _article("gid-1", "Clarin.com"))
    assert row is not None
    assert row.source == "Clarín"


def test_save_article_groups_duplicate_variants(db):
    save_article(db, _article("gid-2", "Clarín"))
    save_article(db, _article("gid-3", "Clarin.com"))
    names = {a.source for a in db.query(Article).all()}
    assert names == {"Clarín"}
