"""Duplicate-case merge matcher (pure, no DB)."""
from datetime import date

from fmedia.case_merge import (
    CaseRecord, effective_country, propose_merges, is_placeholder_name,
)


def rec(id, **kw):
    base = dict(country="UK", victim_name=None, victim_age=None,
                incident_location=None, incident_region=None,
                incident_date=None, first_published=None,
                event_type=None, perpetrator_name=None,
                article_count=1, article_incident_countries=())
    base.update(kw)
    return CaseRecord(id=id, **base)


def tiers(groups):
    return {(g.tier, g.survivor_id, g.member_ids) for g in groups}


# ── country attribution ──────────────────────────────────────────────────────
def test_effective_country_majority_incident_country_gb_is_uk():
    r = rec(1, country="DE", article_incident_countries=("GB", "GB", "UK", "DE"))
    assert effective_country(r) == ("UK", True)


def test_effective_country_fallback_to_outlet_when_no_incident_info():
    assert effective_country(rec(1, country="DE")) == ("DE", False)


def test_effective_country_tie_prefers_outlet_country():
    r = rec(1, country="DE", article_incident_countries=("GB", "DE"))
    assert effective_country(r) == ("DE", True)


def test_placeholders():
    for n in (None, "", "Desconocido", "unknown", " N/A ", "Unidentified woman", "sin identificar"):
        assert is_placeholder_name(n)
    assert not is_placeholder_name("Alice Example")


# ── Tier A ───────────────────────────────────────────────────────────────────
def test_cross_country_pair_cross_country_is_tier_a_survivor_most_articles():
    uk = rec(101, country="UK", victim_name="Alice Example", article_count=32,
             incident_date=date(2024, 3, 3), incident_location="Sampletown")
    de = rec(102, country="DE", victim_name="Alice Example", article_count=1,
             incident_date=date(2024, 3, 3), incident_location="Sampletown")
    g, = propose_merges([de, uk])
    assert g.tier == "A" and g.survivor_id == 101 and g.member_ids == (101, 102)
    assert g.country == "UK"


def test_survivor_tie_breaks_on_oldest_id():
    a = rec(9, victim_name="Ana Perez Ruiz", incident_date=date(2024, 1, 1), article_count=2)
    b = rec(4, victim_name="Ana Perez Ruiz", incident_date=date(2024, 1, 2), article_count=2)
    g, = propose_merges([a, b])
    assert g.survivor_id == 4


def test_late_coverage_trial_with_fallback_dates_is_tier_a():
    a = rec(1, victim_name="Joan Smith Brown", first_published=date(2024, 3, 10))
    b = rec(2, victim_name="Joan Smith Brown", first_published=date(2022, 6, 1))
    assert [g.tier for g in propose_merges([a, b])] == ["A"]


def test_same_name_far_apart_exact_dates_not_merged():
    a = rec(1, victim_name="Maria Lopez", incident_date=date(2020, 1, 1))
    b = rec(2, victim_name="Maria Lopez", incident_date=date(2023, 1, 1))
    assert propose_merges([a, b]) == []


def test_single_token_name_never_tier_a():
    a = rec(1, victim_name="Maria", incident_date=date(2024, 1, 1))
    b = rec(2, victim_name="Maria", incident_date=date(2024, 1, 1))
    assert all(g.tier != "A" for g in propose_merges([a, b]))


def test_substring_of_different_name_not_matched():
    a = rec(1, victim_name="Ana Lopez", incident_date=date(2024, 1, 1))
    b = rec(2, victim_name="Diana Lopez", incident_date=date(2024, 1, 1))
    assert propose_merges([a, b]) == []


def test_same_name_different_incident_countries_downgraded_to_b():
    a = rec(1, country="MX", victim_name="Maria Garcia Lopez", incident_date=date(2024, 1, 1),
            article_incident_countries=("MX",))
    b = rec(2, country="ES", victim_name="Maria Garcia Lopez", incident_date=date(2024, 1, 3),
            article_incident_countries=("ES",))
    assert [g.tier for g in propose_merges([a, b])] == ["B"]


def test_name_without_any_date_is_tier_b():
    a = rec(1, victim_name="Alice Example")
    b = rec(2, victim_name="Alice Example")
    assert [g.tier for g in propose_merges([a, b])] == ["B"]


# ── hard guards ──────────────────────────────────────────────────────────────
def test_age_gap_over_two_blocks_everything():
    a = rec(1, victim_name="Jane Doe Roe", victim_age=30, incident_date=date(2024, 1, 1))
    b = rec(2, victim_name="Jane Doe Roe", victim_age=33, incident_date=date(2024, 1, 1))
    assert propose_merges([a, b]) == []
    c = rec(3, victim_age=30, incident_location="Leeds", incident_date=date(2024, 5, 1))
    d = rec(4, victim_age=40, incident_location="Leeds", incident_date=date(2024, 5, 2))
    assert propose_merges([c, d]) == []


def test_age_gap_of_two_allowed():
    a = rec(1, victim_name="Jane Doe Roe", victim_age=30, incident_date=date(2024, 1, 1))
    b = rec(2, victim_name="Jane Doe Roe", victim_age=32, incident_date=date(2024, 1, 1))
    assert len(propose_merges([a, b])) == 1


def test_fatal_vs_nonfatal_never_merged():
    a = rec(1, victim_name="Jane Doe Roe", event_type="category_a", incident_date=date(2024, 1, 1))
    b = rec(2, victim_name="Jane Doe Roe", event_type="category_d", incident_date=date(2024, 1, 1))
    assert propose_merges([a, b]) == []


# ── Tier B ───────────────────────────────────────────────────────────────────
def test_unnamed_same_town_close_dates_is_tier_b():
    a = rec(1, victim_name="Desconocido", incident_location="Leeds", incident_date=date(2024, 5, 1))
    b = rec(2, incident_location="leeds ", incident_date=date(2024, 5, 6))
    g, = propose_merges([a, b])
    assert g.tier == "B"


def test_tier_b_boundary_7_days_in_8_out():
    a = rec(1, incident_location="Leeds", incident_date=date(2024, 5, 1))
    assert [g.tier for g in propose_merges([a, rec(2, incident_location="Leeds", incident_date=date(2024, 5, 8))])] == ["B"]
    out = propose_merges([a, rec(3, incident_location="Leeds", incident_date=date(2024, 5, 9))])
    assert [g.tier for g in out] == ["C"]  # same year -> weak


def test_tier_b_requires_same_country_and_specific_town():
    a = rec(1, country="UK", incident_location="Leeds", incident_date=date(2024, 5, 1))
    b = rec(2, country="CL", incident_location="Leeds", incident_date=date(2024, 5, 1))
    assert propose_merges([a, b]) == []
    c = rec(3, incident_location="England", incident_date=date(2024, 5, 1))
    d = rec(4, incident_location="England", incident_date=date(2024, 5, 1))
    assert propose_merges([c, d]) == []


def test_two_different_named_victims_same_town_never_merged():
    a = rec(1, victim_name="Ana Lopez", incident_location="Leeds", incident_date=date(2024, 5, 1))
    b = rec(2, victim_name="Eva Smith", incident_location="Leeds", incident_date=date(2024, 5, 1))
    assert propose_merges([a, b]) == []


def test_tier_b_type_conflict_known_both_blocks_b():
    a = rec(1, event_type="category_a", incident_location="Leeds", incident_date=date(2024, 5, 1))
    b = rec(2, event_type="category_b", incident_location="Leeds", incident_date=date(2024, 5, 1))
    assert all(g.tier != "B" for g in propose_merges([a, b]))


# ── Tier C ───────────────────────────────────────────────────────────────────
def test_tier_c_same_town_same_year_pairs_only():
    a = rec(1, incident_location="Leeds", incident_date=date(2024, 1, 5))
    b = rec(2, incident_location="Leeds", incident_date=date(2024, 11, 5))
    g, = propose_merges([a, b])
    assert g.tier == "C" and g.member_ids == (1, 2)


def test_tier_c_different_year_nothing():
    a = rec(1, incident_location="Leeds", incident_date=date(2023, 6, 30))
    b = rec(2, incident_location="Leeds", incident_date=date(2024, 1, 2))
    assert propose_merges([a, b]) == []


# ── grouping ────────────────────────────────────────────────────────────────
def test_three_way_group_and_deterministic_order():
    cs = [rec(i, victim_name="Alice Example", incident_date=date(2024, 3, 3), article_count=n)
          for i, n in ((5, 1), (3, 10), (8, 2))]
    g1 = propose_merges(cs)
    g2 = propose_merges(list(reversed(cs)))
    assert g1 == g2 and len(g1) == 1
    assert g1[0].survivor_id == 3 and g1[0].member_ids == (3, 5, 8)


def test_chaining_blocked_when_members_conflict():
    # 1~2 by age ok, 2~3 ok, but 1 vs 3 age gap >2 -> 3 must not join
    a = rec(1, victim_name="Jane Doe Roe", victim_age=30, incident_date=date(2024, 1, 1))
    b = rec(2, victim_name="Jane Doe Roe", victim_age=None, incident_date=date(2024, 1, 1))
    c = rec(3, victim_name="Jane Doe Roe", victim_age=40, incident_date=date(2024, 1, 1))
    gs = propose_merges([a, b, c])
    assert all(not ({1, 3} <= set(g.member_ids)) for g in gs)


# ── placeless same-day unnamed (Christmas example) ─────────────
XMAS = date(2024, 12, 25)


def placeless(id, text, **kw):
    return rec(id, incident_date=XMAS, weapon="arma_blanca", event_type="category_c",
               texts=(text,), article_incident_countries=("GB",), **kw)


def test_placeless_with_text_evidence_is_tier_b():
    a = placeless(18285, "A man appeared in court accused of stabbing two women to death with a kitchen knife on Christmas Day, while an injured boy fled")
    b = placeless(18287, "A 38-year-old woman was fatally stabbed in a knife attack on Christmas Day, alongside a 24-year-old victim and a family dog", victim_age=38)
    g, = propose_merges([a, b])
    assert g.tier == "B" and "placeless" in g.reasons[0]


def test_placeless_date_and_weapon_alone_never_merges():
    a = placeless(1, "Woman killed in flat fire inquiry opened")
    b = placeless(2, "Teenager charged after incident near station")
    gs = propose_merges([a, b])
    assert [g.tier for g in gs] == ["C"]


def test_placeless_different_weapon_or_country_not_linked():
    a = placeless(1, "stabbed knife christmas woman house")
    b = rec(2, incident_date=XMAS, weapon="arma_de_fuego", texts=("stabbed knife christmas woman house",),
            article_incident_countries=("GB",), event_type="category_c")
    c = rec(3, country="CL", incident_date=XMAS, weapon="arma_blanca", texts=("stabbed knife christmas woman house",),
            event_type="category_c")
    assert propose_merges([a, b, c]) == []


def test_impossible_incident_date_is_ignored_for_placeless():
    # 2017 coverage but incident_date extracted as 2024-12-25 -> not a real date
    bad = placeless(1, "stabbed knife christmas woman mother", first_published=date(2017, 12, 28))
    good = placeless(2, "stabbed knife christmas woman mother", first_published=date(2024, 12, 27))
    assert propose_merges([bad, good]) == []


def test_unnamed_chain_longer_than_window_is_not_one_group():
    cs = [rec(i, incident_location="Leeds", incident_date=date(2024, 5, 1 + 6 * (i - 1))) for i in (1, 2, 3)]
    gs = [g for g in propose_merges(cs) if g.tier == "B"]
    assert all(len(g.member_ids) == 2 for g in gs)


# ── Precision fixes ──────────────
def test_fuzzy_name_xaviera_vs_javiera_is_not_tier_a():
    a = rec(536, country="CL", victim_name="Xaviera Ejemplo", first_published=date(2024, 1, 4))
    b = rec(2050, country="UK", victim_name="Javiera Ejemplo", first_published=date(2024, 12, 2))
    assert all(g.tier != "A" for g in propose_merges([a, b]))


def test_common_given_names_subset_is_only_tier_b():
    a = rec(1593, country="MX", victim_name="Ana María", first_published=date(2019, 9, 27))
    b = rec(9707, country="AR", victim_name="Ana María Fictional", first_published=date(2020, 7, 22))
    assert [g.tier for g in propose_merges([a, b])] == ["B"]


def test_subset_with_a_surname_is_still_tier_a():
    a = rec(1, victim_name="Alana Fictional", first_published=date(2025, 10, 31))
    b = rec(2, victim_name="Alana Fictional Smith", first_published=date(2026, 3, 26))
    assert [g.tier for g in propose_merges([a, b])] == ["A"]


def test_german_and_spanish_placeholders():
    for n in ("Unbekannt", "unbekannte Frau", "No especificado", "una mujer no identificada", "Febronio N"):
        assert is_placeholder_name(n) or not __import__("fmedia.case_merge", fromlist=["is_full_name"]).is_full_name(n)


def test_big_city_unnamed_needs_same_day_and_text_evidence():
    a = rec(17179, incident_location="London", first_published=date(2018, 6, 1),
            texts=("Woman dead and six injured in separate London attacks",))
    b = rec(17190, incident_location="London", first_published=date(2018, 5, 25),
            texts=("Man, 95, arrested on suspicion of murdering care worker in London",))
    assert all(g.tier != "B" for g in propose_merges([a, b]))
    c = rec(1, incident_location="London", first_published=date(2012, 4, 12),
            texts=("Casino woman stabbed to death in London penthouse",))
    d = rec(2, incident_location="London", first_published=date(2012, 4, 12),
            texts=("Crack-addict serial rapist stabbed mother to death in front of her son",))
    assert all(g.tier != "B" for g in propose_merges([c, d]))
    e = rec(3, incident_location="London", first_published=date(2024, 9, 9),
            texts=("Man, 20, in court charged with fatal stabbing of woman at a local street festival",))
    f = rec(4, incident_location="London", first_published=date(2024, 9, 9),
            texts=("Brothers in court after mum stabbed to death at a local street festival in front of daughter",))
    assert [g.tier for g in propose_merges([e, f])] == ["B"]


def test_small_town_unnamed_rule_unchanged():
    a = rec(1, incident_location="Malloa", first_published=date(2025, 9, 26), victim_age=62)
    b = rec(2, incident_location="Malloa", first_published=date(2025, 9, 27), victim_age=62)
    assert [g.tier for g in propose_merges([a, b])] == ["B"]


def test_same_perpetrator_full_name_is_tier_a_within_window():
    a = rec(4977, perpetrator_name="Peter Placeholder", first_published=date(2024, 7, 10))
    b = rec(5200, perpetrator_name="Peter Placeholder", first_published=date(2025, 1, 22), victim_name="Laura Sample")
    g, = propose_merges([a, b])
    assert g.tier == "A" and "perpetrator" in g.reasons[0]


def test_same_perpetrator_different_victim_names_never_merged():
    a = rec(1, perpetrator_name="Ivan Sample", victim_name="Helen Testcase", first_published=date(2016, 7, 19))
    b = rec(2, perpetrator_name="Ivan Sample", victim_name="Diana Testcase", first_published=date(2016, 9, 1))
    assert propose_merges([a, b]) == []


def test_perpetrator_placeholder_or_far_apart_not_tier_a():
    a = rec(1, perpetrator_name="husband", first_published=date(2024, 1, 1))
    b = rec(2, perpetrator_name="husband", first_published=date(2024, 1, 2))
    assert propose_merges([a, b]) == []
    c = rec(3, perpetrator_name="Damian Placeholder", first_published=date(2024, 9, 20))
    d = rec(4, perpetrator_name="Damian Placeholder", first_published=date(2023, 1, 20))
    assert [g.tier for g in propose_merges([c, d])] == ["B"]


def test_perpetrator_of_only_common_given_names_is_not_an_identity():
    a = rec(1, country="MX", perpetrator_name="Luis Ángel", first_published=date(2018, 1, 10))
    b = rec(2, country="MX", perpetrator_name="Luis Ángel", first_published=date(2017, 7, 13))
    assert propose_merges([a, b]) == []


def test_big_city_generic_vocabulary_is_not_text_evidence():
    a = rec(16910, incident_location="London", first_published=date(2012, 4, 12),
            texts=("Casino woman stabbed to death in London penthouse",))
    b = rec(16915, incident_location="London", first_published=date(2012, 4, 12),
            texts=("Crack-addict serial rapist stabbed mother to death in front of her son after being freed on bail",))
    assert all(g.tier != "B" for g in propose_merges([a, b]))
