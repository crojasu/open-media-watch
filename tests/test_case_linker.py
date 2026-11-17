"""Ingestion-time linker v2 (fmedia/case_linker.py) through the
public entry point cases.link_article_to_case, SQLite, no LLM."""
from datetime import date

import pytest

from fmedia import case_linker
from fmedia.case_linker import (
    article_incident_country, create_case_from_article_v2, find_matching_case_v2,
    plausible_incident_date, titles_match,
)
from fmedia.cases import link_article_to_case
from fmedia.database import Article, Case

_n = [0]


def art(db, **kw):
    _n[0] += 1
    d = dict(guardian_id=f"lk-{_n[0]}", title="Woman killed", published_date=date(2024, 7, 10), country="UK",
             review_status="confirmed", event_type="category_a")
    d.update(kw)
    a = Article(**d)
    db.add(a)
    db.commit()
    db.refresh(a)
    return a


def link(db, a):
    cid = link_article_to_case(db, a, Case)
    db.commit()
    return cid


@pytest.fixture(autouse=True)
def _v2_on(monkeypatch):
    monkeypatch.delenv(case_linker.V2_ENV, raising=False)
    monkeypatch.delenv("AGENT_LINK_ENABLED", raising=False)


# ── attribution ──────────────────────────────────────────────────────────────
def test_case_country_is_incident_country_not_outlet(db):
    a = art(db, country="UK", incident_country="CO", victim_name="Alice Example", incident_location="Capital City",
            title="Girl's death shocks the country", published_date=date(2019, 5, 4))
    cid = link(db, a)
    c = db.get(Case, cid)
    assert cid is not None and c.country == "CO"
    db.refresh(a)
    assert a.case_id == cid


def test_gb_incident_country_maps_to_uk_and_null_falls_back_to_outlet(db):
    a = art(db, incident_country="GB")
    b = art(db, incident_country=None, country="DE", title="Frau in Hamburg getötet", incident_location="Hamburg")
    assert db.get(Case, link(db, a)).country == "UK"
    assert db.get(Case, link(db, b)).country == "DE"


def test_bogota_impossible_incident_date_is_dropped(db):
    a = art(db, published_date=date(2019, 5, 4), incident_date_extracted=date(2024, 12, 4), incident_country="CO")
    assert plausible_incident_date(a) is None
    assert create_case_from_article_v2(a)["incident_date"] is None


def test_kill_switch_restores_legacy_foreign_guard(db, monkeypatch):
    monkeypatch.setenv(case_linker.V2_ENV, "false")
    a = art(db, country="UK", incident_country="IN")
    assert link(db, a) is None and db.query(Case).count() == 0
    monkeypatch.setenv(case_linker.V2_ENV, "true")
    assert link(db, a) is not None and db.get(Case, a.case_id).country == "IN"


# ── maintainer's rule 1: same full name across outlet countries ───────────────────
def test_cross_country_pair_pair_de_outlet_joins_uk_case(db):
    uk = art(db, country="UK", incident_country="GB", victim_name="Alice Example", incident_location="Sampletown",
             title="Alice Example: Man arrested on suspicion of murder", published_date=date(2024, 3, 10))
    uk_cid = link(db, uk)
    de = art(db, country="DE", victim_name="Alice Example", incident_location="Sampletown",
             title="Mord an Alice Example: Neue Aussagen", published_date=date(2024, 3, 19))
    assert link(db, de) == uk_cid
    c = db.get(Case, uk_cid)
    assert c.country == "UK" and c.article_count == 2
    # six months later, the sentencing (fallback dates, 730-day window): same case
    late = art(db, country="DE", victim_name="Alice Example", title="Urteil im Mordfall Alice Example",
               published_date=date(2024, 9, 30))
    assert link(db, late) == uk_cid


def test_substring_name_tia_vs_tatiana_never_joins(db):
    a = art(db, country="UK", incident_country="CO", victim_name="Tatiana Fictional", published_date=date(2024, 7, 30),
            title="Armed groups target women in the region")
    cid = link(db, a)
    b = art(db, country="UK", victim_name="Tia Fictional", incident_location="Sample Hill", published_date=date(2024, 10, 14),
            title="Sample Hill: Sam Sample admits murder of wife Tia")
    assert link(db, b) != cid


def test_different_full_names_same_town_same_day_stay_apart(db):
    a = art(db, victim_name="Ana Lopez", incident_location="Leeds", title="Ana Lopez killed in Leeds")
    b = art(db, victim_name="Eva Smith", incident_location="Leeds", title="Eva Smith killed in Leeds")
    assert link(db, a) != link(db, b)


# ── perpetrator name ─────────────────────────────────────────────────────────
def test_same_perpetrator_name_joins_case(db):
    a = art(db, perpetrator_name="Peter Placeholder", title="Suspect Peter Placeholder arrested over murders", published_date=date(2024, 7, 12))
    cid = link(db, a)
    b = art(db, perpetrator_name="Peter Placeholder", victim_name="Laura Sample", incident_location="Bushey",
            title="Bushey crossbow attack: man admits killing mum and two daughters", published_date=date(2025, 1, 22))
    assert link(db, b) == cid


# ── maintainer's rule 2: unnamed + same town + close dates ────────────────────────
def test_two_unknown_victims_same_town_within_week_join(db):
    a = art(db, victim_name="Unknown", incident_location="Talca", title="Incidente en centro de Talca",
            country="CL", incident_country="CL", published_date=date(2025, 8, 30))
    cid = link(db, a)
    b = art(db, victim_name="No especificado", incident_location="Talca", title="Decretan prisión preventiva en Talca",
            country="CL", incident_country="CL", published_date=date(2025, 9, 4))
    assert link(db, b) == cid
    assert db.get(Case, cid).victim_name is None  # placeholders are never stored as a name


def test_unknown_victims_same_town_eight_days_apart_do_not_join(db):
    a = art(db, victim_name="Unknown", incident_location="Talca", country="CL", published_date=date(2025, 8, 30))
    b = art(db, victim_name="Unknown", incident_location="Talca", country="CL", published_date=date(2025, 9, 8),
            title="Otro caso en Talca")
    assert link(db, a) != link(db, b)


def test_big_city_unnamed_needs_text_evidence(db):
    a = art(db, incident_location="London", title="Woman dead and six injured in separate London attacks",
            published_date=date(2018, 6, 1))
    b = art(db, incident_location="London", title="Man, 95, arrested on suspicion of murdering care worker in London",
            published_date=date(2018, 5, 31))
    assert link(db, a) != link(db, b)


# ── placeless same-day (Christmas 2024) ──────────────────────────────────────
XMAS_TITLES = [
    "Christmas Day killer in court after stabbing two women to death with kitchen knife as injured boy fled",
    "Pictured: Christmas Day murder victim, 24, killed in knife attack that also claimed life of 38-year-old woman",
    "Man, 49, is charged with murder after two women, 24 and 38, are killed in Christmas Day knife attack",
    "Dad arrested after women, 24 and 38, including his partner killed as screams heard in Xmas Day stabbing rampage",
    "Women knifed to death in Xmas Day rampage as desperate boy yelled he's going to kill me while trying to flee",
]


def test_christmas_double_stabbing_is_one_case(db):
    ids = set()
    for i, t in enumerate(XMAS_TITLES):
        a = art(db, title=t, incident_date_extracted=date(2024, 12, 25), weapon="arma_blanca",
                event_type="category_a" if i == 3 else "category_c",
                published_date=date(2024, 12, 27 + (i % 2)), incident_country="GB")
        ids.add(link(db, a))
    assert len(ids) == 1
    assert db.get(Case, ids.pop()).article_count == 5


def test_placeless_same_day_without_text_evidence_stays_apart(db):
    a = art(db, title="Woman killed in flat fire inquiry opened", incident_date_extracted=date(2024, 12, 25),
            weapon="arma_blanca", published_date=date(2024, 12, 27))
    b = art(db, title="Teenager charged after incident near station", incident_date_extracted=date(2024, 12, 25),
            weapon="arma_blanca", published_date=date(2024, 12, 27))
    assert link(db, a) != link(db, b)


# ── syndicated / near-identical titles ───────────────────────────────────────
def test_identical_title_other_outlet_joins_even_without_name_or_place(db):
    a = art(db, title="Man arrested on suspicion of murder after woman, 31, found stabbed to death in home",
            source="The Independent")
    cid = link(db, a)
    b = art(db, title="Man arrested on suspicion of murder after woman, 31, found stabbed to death in home",
            source="independent.co.uk", published_date=date(2024, 7, 11))
    assert link(db, b) == cid


def test_generic_template_titles_do_not_join():
    assert not titles_match("Man charged with murder after woman found dead at home in north Wales",
                            "Man charged with murder after woman found fatally stabbed in Exampleton")
    assert titles_match("CCTV shows killer buying cable ties hours before he strangled woman to death in her home",
                        "Killer caught on CCTV buying cable ties before strangling woman to death")


def test_redditch_and_north_wales_are_two_cases(db):
    a = art(db, title="Man charged with murder after woman found dead at home in north Wales",
            incident_location="north Wales", published_date=date(2024, 11, 7))
    b = art(db, title="Man charged with murder after woman found fatally stabbed in Exampleton",
            incident_location="Exampleton", published_date=date(2024, 11, 9))
    assert link(db, a) != link(db, b)


# ── follow-up coverage of an old incident ────────────────────────────────────
def test_trial_article_years_later_joins_named_case_not_new_case(db):
    first = art(db, victim_name="Helen Testcase", incident_location="Hertfordshire", published_date=date(2016, 7, 19),
                title="Helen Testcase death: author found in septic tank")
    cid = link(db, first)
    trial = art(db, victim_name="Helen Testcase", published_date=date(2017, 2, 23),
                title="Helen Testcase murder: Ivan Sample jailed for life")
    assert link(db, trial) == cid
    assert db.query(Case).count() == 1


def test_fatal_and_attempt_never_share_a_case(db):
    a = art(db, victim_name="Jane Doe Roe", event_type="category_a")
    b = art(db, victim_name="Jane Doe Roe", event_type="category_d", title="Jane Doe Roe survives attack")
    assert link(db, a) != link(db, b)


# ── tombstones ───────────────────────────────────────────────────────────────
def test_match_on_a_merged_case_redirects_to_survivor(db):
    a = art(db, victim_name="Nora Testcase", published_date=date(2024, 2, 23))
    old = link(db, a)
    survivor = Case(country="UK", victim_name="Nora Testcase", article_count=3)
    db.add(survivor)
    db.commit()
    db.get(Case, old).merged_into_id = survivor.id
    db.commit()
    b = art(db, victim_name="Nora Testcase", published_date=date(2024, 4, 5), title="Nora Testcase: body cut into pieces")
    assert link(db, b) == survivor.id


def test_article_incident_country_helper():
    class A:  # noqa
        country, incident_country = "DE", "GB"
    assert article_incident_country(A()) == "UK"
    A.incident_country = None
    assert article_incident_country(A()) == "DE"
