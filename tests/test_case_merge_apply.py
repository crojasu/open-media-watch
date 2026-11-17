"""Reversible merge application (fmedia/case_merge_apply.py) and
the CLI guards of scripts/case_merge_apply.py."""
from datetime import date, datetime

import pytest

from fmedia.case_merge import MergeGroup, propose_merges
from fmedia.case_merge_apply import (
    apply_groups, load_case_records, merge_group, resolve_case_id, unmerge_batch,
)
from fmedia.database import Article, Case, CaseMergeAudit

_n = [0]


def mk_case(db, **kw):
    d = dict(country="UK", article_count=1)
    d.update(kw)
    c = Case(**d)
    db.add(c)
    db.commit()
    db.refresh(c)
    return c


def mk_art(db, case, **kw):
    _n[0] += 1
    d = dict(guardian_id=f"ma-{_n[0]}", title=f"t{_n[0]}", published_date=date(2024, 3, 10), country=case.country,
             review_status="confirmed", event_type="category_c", case_id=case.id)
    d.update(kw)
    a = Article(**d)
    db.add(a)
    db.commit()
    return a


def cross_country_pair(db):
    uk = mk_case(db, country="UK", victim_name="Alice Example", victim_age=33, incident_location="Sampletown",
                 incident_date=date(2024, 3, 3), article_count=2, perpetrator_name="Peter Placeholder",
                 first_reported_at=datetime(2024, 3, 10))
    de = mk_case(db, country="DE", victim_name="Alice Example", incident_date=date(2024, 3, 3), weapon="weapon_x",
                 first_reported_at=datetime(2024, 3, 19))
    mk_art(db, uk, incident_country="GB")
    mk_art(db, uk, incident_country="GB", published_date=date(2024, 3, 16))
    mk_art(db, de, country="DE", published_date=date(2024, 3, 19))
    return uk, de


def test_load_records_and_propose_cross_country_pair_group(db):
    uk, de = cross_country_pair(db)
    recs = load_case_records(db)
    g, = propose_merges(recs)
    assert g.tier == "A" and g.survivor_id == uk.id and g.member_ids == (uk.id, de.id) and g.country == "UK"


def test_merge_moves_articles_enriches_and_tombstones(db):
    uk, de = cross_country_pair(db)
    g, = propose_merges(load_case_records(db))
    audits = merge_group(db, g, "batch-1")
    db.commit()
    uk, de = db.get(Case, uk.id), db.get(Case, de.id)
    assert de.merged_into_id == uk.id and uk.merged_into_id is None
    assert db.query(Article).filter(Article.case_id == uk.id).count() == 3
    assert db.query(Article).filter(Article.case_id == de.id).count() == 0
    assert uk.article_count == 3 and uk.weapon == "weapon_x" and uk.country == "UK"
    assert len(audits) == 1 and audits[0].batch_id == "batch-1" and audits[0].tier == "A"
    assert resolve_case_id(db, de.id) == uk.id


def test_unmerge_round_trip_restores_everything(db):
    uk, de = cross_country_pair(db)
    before_uk = {k: getattr(uk, k) for k in ("victim_age", "weapon", "article_count", "country", "incident_location")}
    before_de_arts = sorted(a.id for a in db.query(Article).filter(Article.case_id == de.id))
    g, = propose_merges(load_case_records(db))
    apply_groups(db, [g], "batch-2", tiers=("A",))
    db.commit()
    assert unmerge_batch(db, "batch-2") == 1
    db.commit()
    uk, de = db.get(Case, uk.id), db.get(Case, de.id)
    assert de.merged_into_id is None
    assert sorted(a.id for a in db.query(Article).filter(Article.case_id == de.id)) == before_de_arts
    assert {k: getattr(uk, k) for k in before_uk} == before_uk
    assert db.query(CaseMergeAudit).filter(CaseMergeAudit.reverted_at.isnot(None)).count() == 1
    assert unmerge_batch(db, "batch-2") == 0  # idempotent


def test_apply_groups_only_requested_tiers(db):
    a = mk_case(db, incident_location="Leeds", first_reported_at=datetime(2024, 5, 1))
    b = mk_case(db, incident_location="Leeds", first_reported_at=datetime(2024, 5, 3))
    mk_art(db, a, published_date=date(2024, 5, 1))
    mk_art(db, b, published_date=date(2024, 5, 3))
    groups = propose_merges(load_case_records(db))
    assert [g.tier for g in groups] == ["B"]
    assert apply_groups(db, groups, "b-only", tiers=("A",)) == {}
    assert db.get(Case, b.id).merged_into_id is None
    stats = apply_groups(db, groups, "b-too", tiers=("A", "B"))
    db.commit()
    assert stats["absorbed"] == 1 and db.get(Case, b.id).merged_into_id == a.id


def test_merge_refuses_already_merged_member(db):
    uk, de = cross_country_pair(db)
    g, = propose_merges(load_case_records(db))
    merge_group(db, g, "x")
    db.commit()
    with pytest.raises(ValueError):
        merge_group(db, MergeGroup(tier="A", survivor_id=uk.id, member_ids=(uk.id, de.id), country="UK"), "y")


def test_load_records_skips_tombstones(db):
    uk, de = cross_country_pair(db)
    de.merged_into_id = uk.id
    db.commit()
    assert [r.id for r in load_case_records(db)] == [uk.id]




# ── CLI guards (no DB access, validate() only) ───────────────────────────────


def test_alembic_migration_files_import():
    import importlib.util, pathlib
    for name in ("010_create_case_merge_audit.py", "011_add_cases_merged_into_id.py"):
        p = pathlib.Path(__file__).resolve().parents[1] / "alembic" / "versions" / name
        spec = importlib.util.spec_from_file_location(name[:-3], p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert callable(mod.upgrade) and callable(mod.downgrade)
    assert mod.down_revision == "010"
