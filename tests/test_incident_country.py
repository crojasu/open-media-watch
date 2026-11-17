"""Wrong-country incident handling.

Bug: a media outlet monitored under `articles.country` (e.g. "UK") can report
a relevant incident that actually happened in a DIFFERENT country (e.g. India). The
classifier had no field for where the incident occurred, so foreign incidents
were silently counted as local. This file covers:

  1. `articles.incident_country` column exists (SQLite, via create_all()).
  2. Classifier persists/parses `incident_country` (mocked LLM call).
  3. `link_article_to_case` (cases.py) refuses to link/create a local Case for
     a foreign-incident article.
  4. `local_incident_filter()` (database.py) excludes foreign, includes NULL
     and same-country.
  5. Every listed view surface (stats query functions, /api/articles)
     excludes foreign incidents but keeps NULL/local ones.
"""
from datetime import date, timedelta
from unittest.mock import MagicMock

import pytest

from fmedia.database import (
    Article, Case, CaseLinkAudit, local_incident_filter,
    get_total_count, get_type_counts, get_articles_paginated,
    get_articles_by_status, get_articles_by_date, get_monthly_counts,
    get_date_range, get_queue_stats,
)
from fmedia.cases import link_article_to_case
from fmedia import llm_classifier

AUTH = ("tester", "testpass")


def _article(db, **overrides):
    defaults = dict(
        guardian_id=overrides.pop("guardian_id", "ic-test"),
        title="Incidente en Valparaíso",
        content_snippet="Una mujer fue asesinada por su expareja",
        published_date=date(2026, 3, 1),
        country="UK",
        review_status="confirmed",
        event_type="category_a",
    )
    defaults.update(overrides)
    art = Article(**defaults)
    db.add(art)
    db.commit()
    db.refresh(art)
    return art


# ── 1. Column exists ──────────────────────────────────────────────────────────

def test_article_model_has_incident_country_column():
    assert hasattr(Article, "incident_country")


def test_incident_country_column_persists_in_sqlite(db):
    art = _article(db, guardian_id="col-1", incident_country="IN")
    db.refresh(art)
    assert art.incident_country == "IN"


def test_incident_country_defaults_to_null(db):
    art = _article(db, guardian_id="col-2")
    assert art.incident_country is None


# ── 2. local_incident_filter() helper ─────────────────────────────────────────

def test_local_incident_filter_includes_null(db):
    _article(db, guardian_id="f-null", incident_country=None)
    ids = [a.id for a in db.query(Article).filter(local_incident_filter()).all()]
    assert len(ids) == 1


def test_local_incident_filter_includes_same_country(db):
    _article(db, guardian_id="f-same", country="CL", incident_country="CL")
    ids = [a.id for a in db.query(Article).filter(local_incident_filter()).all()]
    assert len(ids) == 1


def test_local_incident_filter_excludes_foreign(db):
    _article(db, guardian_id="f-foreign", country="UK", incident_country="IN")
    ids = [a.id for a in db.query(Article).filter(local_incident_filter()).all()]
    assert ids == []


def test_local_incident_filter_mixed_set(db):
    local_null = _article(db, guardian_id="mix-1", country="UK", incident_country=None)
    local_same = _article(db, guardian_id="mix-2", country="UK", incident_country="UK")
    foreign = _article(db, guardian_id="mix-3", country="UK", incident_country="IN")
    ids = {a.id for a in db.query(Article).filter(local_incident_filter()).all()}
    assert ids == {local_null.id, local_same.id}
    assert foreign.id not in ids


# ── 2b. UK ("UK") vs the classifier's real ISO-2 for the UK ("GB") ────────────
# `articles.country` uses the app's internal codes, where the UK is "UK"; the
# classifier writes real ISO-3166 alpha-2, so a UK incident arrives as "GB".
# Both spellings must read as LOCAL: dropping either silently hides thousands
# of confirmed UK cases (the /case/{id} 404 bug).

def test_local_incident_filter_includes_uk_incident_as_gb(db):
    art = _article(db, guardian_id="gb-1", country="UK", incident_country="GB")
    ids = [a.id for a in db.query(Article).filter(local_incident_filter()).all()]
    assert ids == [art.id]


def test_local_incident_filter_accepts_both_uk_spellings(db):
    iso = _article(db, guardian_id="gb-2", country="UK", incident_country="GB")
    legacy = _article(db, guardian_id="gb-3", country="UK", incident_country="UK")
    foreign = _article(db, guardian_id="gb-4", country="UK", incident_country="IN")
    ids = {a.id for a in db.query(Article).filter(local_incident_filter()).all()}
    assert ids == {iso.id, legacy.id}
    assert foreign.id not in ids


def test_local_incident_filter_excludes_gb_incident_for_other_country(db):
    # Normalization is per-row: a Spanish outlet reporting a UK incident stays foreign.
    _article(db, guardian_id="gb-5", country="ES", incident_country="GB")
    ids = [a.id for a in db.query(Article).filter(local_incident_filter()).all()]
    assert ids == []


def test_link_article_to_case_allows_uk_incident_as_gb(db):
    art = _article(db, guardian_id="gb-6", country="UK", incident_country="GB",
                   victim_name="Jane Doe")
    assert link_article_to_case(db, art, Case) is not None


# ── 3. Classifier parses/persists incident_country ────────────────────────────

def _fake_llm_result(**overrides):
    result = dict(
        type="category_a", relacion="intimo_ex", motivo="rechazo_separacion",
        confidence=0.95, reason="test", provider="claude/claude-haiku-4-5-20251001",
        tokens_in=100, tokens_out=50, cache_write_tokens=0, cache_read_tokens=0,
        victim_name="Ana Pérez", incident_country="IN",
    )
    result.update(overrides)
    return result


def test_classify_one_persists_valid_incident_country(db, monkeypatch):
    monkeypatch.setattr(llm_classifier, "classify_article", lambda *a, **k: _fake_llm_result())
    art = _article(db, guardian_id="clf-1", country="UK", event_type=None, review_status="pending")
    counts = {t: 0 for t in list(llm_classifier.POSITIVE_TYPE_IDS) + ["rejected", "skipped", "errors"]}

    llm_classifier._classify_one(art, 0, 1, counts, db)

    assert art.incident_country == "IN"
    # Still confirmed — the article IS a real relevant incident, just not local (#2 decision rule).
    assert art.review_status == "confirmed"
    assert art.human_label == 1


def test_classify_one_ignores_malformed_incident_country(db, monkeypatch):
    monkeypatch.setattr(
        llm_classifier, "classify_article",
        lambda *a, **k: _fake_llm_result(incident_country="India"),  # not ISO-2
    )
    art = _article(db, guardian_id="clf-2", country="UK", event_type=None, review_status="pending")
    counts = {t: 0 for t in list(llm_classifier.POSITIVE_TYPE_IDS) + ["rejected", "skipped", "errors"]}

    llm_classifier._classify_one(art, 0, 1, counts, db)

    assert art.incident_country is None


def test_classify_one_null_incident_country_stays_null(db, monkeypatch):
    monkeypatch.setattr(
        llm_classifier, "classify_article",
        lambda *a, **k: _fake_llm_result(incident_country=None),
    )
    art = _article(db, guardian_id="clf-3", country="UK", event_type=None, review_status="pending")
    counts = {t: 0 for t in list(llm_classifier.POSITIVE_TYPE_IDS) + ["rejected", "skipped", "errors"]}

    llm_classifier._classify_one(art, 0, 1, counts, db)

    assert art.incident_country is None


def test_classify_one_lowercases_and_normalizes_incident_country(db, monkeypatch):
    monkeypatch.setattr(
        llm_classifier, "classify_article",
        lambda *a, **k: _fake_llm_result(incident_country=" in "),
    )
    art = _article(db, guardian_id="clf-4", country="UK", event_type=None, review_status="pending")
    counts = {t: 0 for t in list(llm_classifier.POSITIVE_TYPE_IDS) + ["rejected", "skipped", "errors"]}

    llm_classifier._classify_one(art, 0, 1, counts, db)

    assert art.incident_country == "IN"


# ── 4. cases.py foreign-incident guard ────────────────────────────────────────

def test_link_article_to_case_foreign_incident_gets_incident_country_case(db):
    # A foreign incident is grouped into a Case of ITS OWN country
    # (never a local one). The legacy guard stays behind the kill switch.
    article = _article(
        db, guardian_id="link-foreign", country="UK", incident_country="IN",
        event_type="category_a",
    )
    case_id = link_article_to_case(db, article, Case)
    db.commit()

    assert case_id is not None
    db.refresh(article)
    assert article.case_id == case_id
    assert db.get(Case, case_id).country == "IN"
    assert db.query(Case).filter(Case.country == "UK").count() == 0


def test_link_article_to_case_refuses_foreign_incident_when_v2_off(db, monkeypatch):
    monkeypatch.setenv("CASE_LINK_V2", "false")
    article = _article(
        db, guardian_id="link-foreign-legacy", country="UK", incident_country="IN",
        event_type="category_a",
    )
    case_id = link_article_to_case(db, article, Case)

    assert case_id is None
    db.refresh(article)
    assert article.case_id is None
    assert db.query(Case).count() == 0


def test_link_article_to_case_still_links_local_article(db):
    article = _article(
        db, guardian_id="link-local", country="UK", incident_country=None,
        event_type="category_a",
    )
    case_id = link_article_to_case(db, article, Case)
    db.commit()

    assert case_id is not None
    db.refresh(article)
    assert article.case_id == case_id


def test_link_article_to_case_links_when_incident_country_matches(db):
    article = _article(
        db, guardian_id="link-match", country="CL", incident_country="CL",
        event_type="category_a",
    )
    case_id = link_article_to_case(db, article, Case)

    assert case_id is not None


def test_link_article_to_case_foreign_guard_runs_before_agent_flag(db, monkeypatch):
    """Legacy linker (CASE_LINK_V2=false): even with AGENT_LINK_ENABLED on, a
    foreign incident must short-circuit before the agent machinery is ever
    imported/invoked."""
    monkeypatch.setenv("CASE_LINK_V2", "false")
    monkeypatch.setenv("AGENT_LINK_ENABLED", "true")
    article = _article(
        db, guardian_id="link-foreign-agent", country="UK", incident_country="IN",
        event_type="category_a",
    )
    case_id = link_article_to_case(db, article, Case)

    assert case_id is None
    assert db.query(CaseLinkAudit).count() == 0  # agent path never ran


def test_link_article_to_case_v2_foreign_incident_with_agent_flag_no_llm(db, monkeypatch):
    """v2 with AGENT_LINK_ENABLED on and no candidates: an IN case is created
    and the agent is never consulted (no audit row, no API call)."""
    monkeypatch.setenv("AGENT_LINK_ENABLED", "true")
    article = _article(
        db, guardian_id="link-foreign-agent-v2", country="UK", incident_country="IN",
        event_type="category_a",
    )
    case_id = link_article_to_case(db, article, Case)
    db.commit()

    assert case_id is not None and db.get(Case, case_id).country == "IN"
    assert db.query(CaseLinkAudit).count() == 0


# ── 5a. database.py query-function surfaces ───────────────────────────────────

def test_get_total_count_excludes_foreign(db):
    _article(db, guardian_id="tc-1", country="UK", incident_country=None)
    _article(db, guardian_id="tc-2", country="UK", incident_country="IN")
    assert get_total_count(db, country="UK") == 1


def test_get_total_count_ignores_incident_country_for_non_confirmed_status(db):
    # Pending/rejected articles aren't "local incident" surfaces at all — the
    # filter is scoped to status == "confirmed" only.
    _article(db, guardian_id="tc-3", country="UK", incident_country="IN", review_status="pending")
    assert get_total_count(db, status="pending", country="UK") == 1


def test_get_type_counts_excludes_foreign(db):
    _article(db, guardian_id="ty-1", country="UK", incident_country=None, event_type="category_a")
    _article(db, guardian_id="ty-2", country="UK", incident_country="IN", event_type="category_a")
    counts = get_type_counts(db, country="UK")
    assert counts.get("category_a") == 1


def test_get_articles_paginated_excludes_foreign_when_confirmed(db):
    _article(db, guardian_id="pg-1", country="UK", incident_country=None)
    _article(db, guardian_id="pg-2", country="UK", incident_country="IN")
    result = get_articles_paginated(db, country="UK")
    assert result["total"] == 1


def test_get_articles_by_status_excludes_foreign_when_confirmed(db):
    _article(db, guardian_id="st-1", country="UK", incident_country=None)
    _article(db, guardian_id="st-2", country="UK", incident_country="IN")
    result = get_articles_by_status(db, status="confirmed", country="UK")
    assert result["total"] == 1


def test_get_articles_by_status_unaffected_for_pending(db):
    _article(db, guardian_id="st-3", country="UK", incident_country="IN", review_status="pending")
    result = get_articles_by_status(db, status="pending", country="UK")
    assert result["total"] == 1


def test_get_articles_by_date_excludes_foreign(db):
    d = date(2026, 5, 1)
    _article(db, guardian_id="dt-1", country="UK", incident_country=None, published_date=d)
    _article(db, guardian_id="dt-2", country="UK", incident_country="IN", published_date=d)
    arts = get_articles_by_date(db, d, country="UK")
    assert len(arts) == 1


def test_get_monthly_counts_excludes_foreign(db):
    d = date(2026, 5, 1)
    _article(db, guardian_id="mo-1", country="UK", incident_country=None, published_date=d)
    _article(db, guardian_id="mo-2", country="UK", incident_country="IN", published_date=d)
    counts = get_monthly_counts(db, country="UK")
    assert sum(c["count"] for c in counts) == 1


def test_get_date_range_excludes_foreign_edges(db):
    _article(db, guardian_id="dr-1", country="UK", incident_country=None, published_date=date(2026, 1, 1))
    _article(db, guardian_id="dr-2", country="UK", incident_country="IN", published_date=date(2020, 1, 1))
    rng = get_date_range(db, country="UK")
    assert rng["first"] == "2026-01-01"
    assert rng["last"] == "2026-01-01"


def test_get_queue_stats_confirmed_excludes_foreign(db):
    _article(db, guardian_id="qs-1", country="UK", incident_country=None)
    _article(db, guardian_id="qs-2", country="UK", incident_country="IN")
    _article(db, guardian_id="qs-3", country="UK", incident_country="IN", review_status="pending")
    stats = get_queue_stats(db, country="UK")
    assert stats["confirmed"] == 1
    assert stats["pending"] == 1  # pending count unaffected by incident_country


# ── 5b. HTTP surfaces: /archive+/api/articles, /case, RSS, /api/* stats ───────

def test_api_articles_excludes_foreign(client, db):
    _article(db, guardian_id="api-1", country="UK", incident_country=None, title="Caso local UK")
    _article(db, guardian_id="api-2", country="UK", incident_country="IN", title="Caso extranjero en UK")
    r = client.get("/api/articles", params={"country": "UK"})
    titles = {a["title"] for a in r.json()["items"]}
    assert titles == {"Caso local UK"}




def test_case_page_404s_for_foreign_incident(client, db):
    art = _article(db, guardian_id="case-foreign", country="UK", incident_country="IN")
    r = client.get(f"/case/{art.id}", auth=AUTH)
    assert r.status_code == 404






def test_api_timeline_excludes_foreign(client, db):
    _article(db, guardian_id="tl-1", country="CL", incident_country=None)
    _article(db, guardian_id="tl-2", country="CL", incident_country="AR")
    r = client.get("/api/timeline", params={"country": "CL"})
    assert r.json()["total"] == 1


def test_api_types_excludes_foreign(client, db):
    _article(db, guardian_id="tp-1", country="CL", incident_country=None, event_type="category_a")
    _article(db, guardian_id="tp-2", country="CL", incident_country="AR", event_type="category_a")
    r = client.get("/api/types", params={"country": "CL"})
    assert r.json()["category_a"] == 1


def test_api_sankey_excludes_foreign(client, db):
    _article(db, guardian_id="sk-1", country="CL", incident_country=None, event_type="category_a")
    _article(db, guardian_id="sk-2", country="CL", incident_country="AR", event_type="category_a")
    r = client.get("/api/sankey", params={"country": "CL"})
    total = sum(row["count"] for row in r.json())
    assert total == 1




# ── 5c. (reserved) ────────────────────────────────────────





# ── 5d. /api/search (search.py) ────────────────────────────────────────────────





# ── 5e. (reserved) ──────────────────────────────────────────────────────────────





