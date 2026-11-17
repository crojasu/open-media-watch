"""
case_linker.py — ingestion-time case linker v2.

Replaces `cases.find_matching_case` / `create_case_from_article` for NEW
articles, behind the kill switch `CASE_LINK_V2` (default on; "false"/"0"/"off"
restores the legacy path byte for byte). The DB-free matching rules live in
fmedia/case_merge.py so that ingestion and the retroactive merge pass can never
disagree about what "the same case" means.

What changed versus the legacy linker (defects found in production):

1. A Case belongs to the INCIDENT country: `Case.country` is the article's
   `incident_country` (GB -> UK) and only falls back to the outlet's country
   when the classifier did not set one. A UK outlet's capital-city story creates a
   CO case; the  guard no longer leaves such articles ungrouped.
2. Same full victim name = one case across outlet countries (maintainer's rule 1),
   with the Tier-A name rules (exact token match, 45 d / 730 d windows).
3. Same full perpetrator name = one case (one named perpetrator was split across 12 cases).
4. Unnamed victim + same specific town + close dates (maintainer's rule 2), with
   the big-city guard; placeholder names ("Unknown", "Unbekannt", "No
   especificado") count as unnamed and are never stored as the victim name.
5. Near-identical title (syndication: one outlet / its own domain, identical
   wire copy) within +-3 days in the same outlet country, deterministic order,
   and a DISTINCTIVE-overlap rule instead of a bare 0.70 ratio ("Man charged
   with murder after woman found dead ..." merged Exampleton with north Wales).
6. Articles with neither a name nor a location are linked too (4,339 confirmed
   articles were never linked because `link_all_unlinked` required one).
"""
from __future__ import annotations

import os
from collections import defaultdict
from datetime import date, datetime, timedelta
from difflib import SequenceMatcher
from typing import Iterable, List, Optional

from sqlalchemy import or_

from . import cases as _legacy
from .case_merge import (
    A_WINDOW_FALLBACK_DAYS, B_WINDOW_DAYS, CaseRecord, _canon_country, _content_words,
    _guards_ok, _name_edge, _perp_edge, _placeless_edge, _tokens, _town, _town_edge,
    is_full_name, is_placeholder_name,
)
from .cases import _normalize, _parse_age, _parse_date

V2_ENV = "CASE_LINK_V2"
TITLE_WINDOW_DAYS = 3
TITLE_CANDIDATE_LIMIT = 500
NAME_CANDIDATE_LIMIT = 400
TITLE_RATIO_MIN = 0.80
TITLE_MIN_SHARED_STEMS = 4
TITLE_MIN_JACCARD = 0.20
MAX_FUTURE_INCIDENT_DAYS = 30  # an incident "after" publication by more than this is an extraction error

from .case_merge import GENERIC_STEMS as _GENERIC_STEMS  # noqa: E402  (shared with the merge matcher)


def v2_enabled() -> bool:
    return os.getenv(V2_ENV, "true").strip().lower() not in ("0", "false", "no", "off")


# ── country / date attribution ───────────────────────────────────────────────

def article_incident_country(article) -> Optional[str]:
    """Internal code of the country where the incident happened (GB -> UK);
    the outlet's country when the classifier did not set `incident_country`."""
    inc = _canon_country(getattr(article, "incident_country", None))
    return inc or _canon_country(getattr(article, "country", None))


def plausible_incident_date(article) -> Optional[date]:
    """`incident_date_extracted`, unless it is impossible (more than
    MAX_FUTURE_INCIDENT_DAYS after publication — seen: a 2016 capital-city story dated 2024-12-04)."""
    idate = _parse_date(getattr(article, "incident_date_extracted", None))
    pub = _parse_date(getattr(article, "published_date", None))
    if idate and pub and (idate - pub).days > MAX_FUTURE_INCIDENT_DAYS:
        return None
    return idate


def clean_victim_name(name: Optional[str]) -> Optional[str]:
    return None if is_placeholder_name(name) else (name or None)


# ── records ──────────────────────────────────────────────────────────────────

def article_record(article) -> CaseRecord:
    """The incoming article as a one-article CaseRecord (id -1 = probe)."""
    pub = _parse_date(getattr(article, "published_date", None))
    return CaseRecord(
        id=-1, country=_canon_country(getattr(article, "country", None)) or "UK",
        victim_name=clean_victim_name(getattr(article, "victim_name", None)),
        victim_age=_parse_age(getattr(article, "victim_age", None)),
        incident_location=getattr(article, "incident_location", None),
        incident_region=getattr(article, "incident_region", None),
        incident_date=plausible_incident_date(article), first_published=pub,
        event_type=getattr(article, "event_type", None),
        perpetrator_name=getattr(article, "perpetrator_name", None),
        article_count=1, weapon=getattr(article, "weapon", None),
        texts=tuple(t for t in (getattr(article, "title", None), getattr(article, "event_summary", None)) if t),
        article_incident_countries=tuple(c for c in (getattr(article, "incident_country", None),) if c),
    )


def _as_date(v) -> Optional[date]:
    if isinstance(v, datetime):
        return v.date()
    return _parse_date(v)


def case_records(db, cases: List, Article) -> List[CaseRecord]:
    """CaseRecords for ORM Case rows, with their articles' incident countries,
    titles and first publication date (one query for the whole batch)."""
    if not cases:
        return []
    ids = [c.id for c in cases]
    inc, titles, first_pub = defaultdict(list), defaultdict(list), {}
    rows = db.query(Article.case_id, Article.incident_country, Article.title, Article.published_date) \
        .filter(Article.case_id.in_(ids)).order_by(Article.published_date, Article.id).all()
    for cid, icc, title, pub in rows:
        if icc:
            inc[cid].append(icc)
        if title and len(titles[cid]) < 5:
            titles[cid].append(title)
        d = _as_date(pub)
        if d and (cid not in first_pub or d < first_pub[cid]):
            first_pub[cid] = d
    out = []
    for c in cases:
        out.append(CaseRecord(
            id=c.id, country=c.country, victim_name=clean_victim_name(c.victim_name), victim_age=c.victim_age,
            incident_location=c.incident_location, incident_region=c.incident_region,
            incident_date=_as_date(c.incident_date),
            first_published=first_pub.get(c.id) or _as_date(c.first_reported_at),
            event_type=c.event_type, perpetrator_name=c.perpetrator_name,
            article_count=c.article_count or 1, weapon=c.weapon,
            texts=tuple(titles.get(c.id, ())) + ((c.summary,) if c.summary else ()),
            article_incident_countries=tuple(inc.get(c.id, ())),
        ))
    return out


# ── title similarity (strategy 5) ────────────────────────────────────────────

def _distinctive_stems(text: str) -> set:
    return {w for w in _content_words(CaseRecord(id=0, country="", texts=(text,))) if w not in _GENERIC_STEMS}


def titles_match(a: Optional[str], b: Optional[str]) -> bool:
    """Same story: near-identical headline, or >= TITLE_MIN_SHARED_STEMS shared
    DISTINCTIVE stems with Jaccard >= TITLE_MIN_JACCARD."""
    if not a or not b:
        return False
    na, nb = _normalize(a), _normalize(b)
    if na == nb:
        return True
    if SequenceMatcher(None, na, nb).ratio() >= TITLE_RATIO_MIN and _distinctive_stems(a) & _distinctive_stems(b):
        return True
    sa, sb = _distinctive_stems(a), _distinctive_stems(b)
    shared = sa & sb
    return len(shared) >= TITLE_MIN_SHARED_STEMS and len(shared) / len(sa | sb) >= TITLE_MIN_JACCARD


# ── candidate retrieval ──────────────────────────────────────────────────────

def _live(q, Case):
    return q.filter(Case.merged_into_id.is_(None))


def _date_window_filter(Case, ref: Optional[date], days: int):
    if ref is None:
        return None
    lo, hi = ref - timedelta(days=days), ref + timedelta(days=days)
    return or_(
        Case.incident_date.between(lo, hi),
        Case.first_reported_at.between(datetime.combine(lo, datetime.min.time()),
                                       datetime.combine(hi, datetime.max.time())),
        Case.incident_date.is_(None) & Case.first_reported_at.is_(None),
    )


def _longest_token(name: Optional[str]) -> Optional[str]:
    toks = sorted((t for t in _tokens(name) if len(t) >= 3), key=len, reverse=True)
    return toks[0] if toks else None


def _name_candidates(db, probe: CaseRecord, Case) -> List:
    if not is_full_name(probe.victim_name):
        return []
    tok = _longest_token(probe.victim_name)
    if not tok:
        return []
    q = _live(db.query(Case).filter(Case.victim_name.isnot(None), Case.victim_name.ilike(f"%{tok}%")), Case)
    w = _date_window_filter(Case, probe.incident_date or probe.first_published, A_WINDOW_FALLBACK_DAYS)
    if w is not None:
        q = q.filter(w)
    return q.order_by(Case.id).limit(NAME_CANDIDATE_LIMIT).all()


def _perp_candidates(db, probe: CaseRecord, Case) -> List:
    tok = _longest_token(probe.perpetrator_name)
    if not tok or not probe.perpetrator_name:
        return []
    q = _live(db.query(Case).filter(Case.perpetrator_name.isnot(None), Case.perpetrator_name.ilike(f"%{tok}%")), Case)
    w = _date_window_filter(Case, probe.incident_date or probe.first_published, A_WINDOW_FALLBACK_DAYS)
    if w is not None:
        q = q.filter(w)
    return q.order_by(Case.id).limit(NAME_CANDIDATE_LIMIT).all()


def _town_candidates(db, probe: CaseRecord, Case, country: str) -> List:
    town = _town(probe)
    ref = probe.incident_date or probe.first_published
    if not town or ref is None:
        return []
    q = _live(db.query(Case).filter(Case.country == country, Case.incident_location.isnot(None),
                                    Case.incident_location.ilike(f"%{town}%")), Case)
    q = q.filter(_date_window_filter(Case, ref, B_WINDOW_DAYS))
    return q.order_by(Case.id).limit(NAME_CANDIDATE_LIMIT).all()


def _placeless_candidates(db, probe: CaseRecord, Case, country: str) -> List:
    if probe.incident_date is None or _town(probe) or not probe.weapon or not is_placeholder_name(probe.victim_name):
        return []
    lo, hi = probe.incident_date - timedelta(days=1), probe.incident_date + timedelta(days=1)
    q = _live(db.query(Case).filter(Case.country == country, Case.incident_date.between(lo, hi),
                                    Case.weapon.isnot(None)), Case)
    return q.order_by(Case.id).limit(NAME_CANDIDATE_LIMIT).all()


def _title_candidate_case_ids(db, article, Article, Case) -> List[int]:
    title = getattr(article, "title", None)
    pub = _parse_date(getattr(article, "published_date", None))
    if not title or pub is None:
        return []
    lo, hi = pub - timedelta(days=TITLE_WINDOW_DAYS), pub + timedelta(days=TITLE_WINDOW_DAYS)
    q = db.query(Article.case_id, Article.title).filter(
        Article.country == getattr(article, "country", None),
        Article.case_id.isnot(None),
        Article.review_status == "confirmed",
        Article.published_date.between(lo, hi),
    )
    own_id = getattr(article, "id", None)
    if own_id is not None:
        q = q.filter(Article.id != own_id)
    rows = q.order_by(Article.published_date, Article.id).limit(TITLE_CANDIDATE_LIMIT).all()
    seen, out = set(), []
    for cid, t in rows:
        if cid in seen:
            continue
        if titles_match(title, t):
            seen.add(cid)
            out.append(cid)
    return out


# ── matching ─────────────────────────────────────────────────────────────────

def _pick(probe: CaseRecord, recs: Iterable[CaseRecord], edge_fn, tiers=("A", "B")) -> Optional[int]:
    best = None
    for r in recs:
        e = edge_fn(probe, r)
        if e and e[0] in tiers and _guards_ok(probe, r):
            key = ("AB".index(e[0]), -(r.article_count or 0), r.id)
            if best is None or key < best[0]:
                best = (key, r.id)
    return best[1] if best else None


def find_matching_case_v2(db, article, Case, Article) -> Optional[object]:
    """Best existing live Case for `article`, or None. Order: victim name,
    perpetrator name, syndicated/near-identical title, unnamed town rule,
    placeless same-day rule. Never links on Tier-C evidence."""
    probe = article_record(article)
    country = article_incident_country(article)
    if not country:
        return None

    recs = case_records(db, _name_candidates(db, probe, Case), Article)
    cid = _pick(probe, recs, _name_edge)
    if cid is None:
        recs = case_records(db, _perp_candidates(db, probe, Case), Article)
        cid = _pick(probe, recs, _perp_edge)
    if cid is None:
        ids = _title_candidate_case_ids(db, article, Article, Case)
        if ids:
            cands = _live(db.query(Case).filter(Case.id.in_(ids)), Case).all()
            for r in case_records(db, cands, Article):
                if _guards_ok(probe, r):
                    cid = r.id
                    break
    if cid is None:
        recs = case_records(db, _town_candidates(db, probe, Case, country), Article)
        cid = _pick(probe, recs, _town_edge)
    if cid is None:
        recs = case_records(db, _placeless_candidates(db, probe, Case, country), Article)
        cid = _pick(probe, recs, _placeless_edge)
    if cid is None:
        return None
    return db.get(Case, resolve_case_id(db, cid, Case))


def resolve_case_id(db, case_id: int, Case) -> int:
    """Follow `merged_into_id` tombstones to the surviving case (bounded)."""
    seen = set()
    while case_id not in seen:
        seen.add(case_id)
        c = db.get(Case, case_id)
        if c is None or c.merged_into_id is None:
            return case_id
        case_id = c.merged_into_id
    return case_id


# ── create / update ──────────────────────────────────────────────────────────

def create_case_from_article_v2(article) -> dict:
    data = _legacy.create_case_from_article(article)
    data["country"] = article_incident_country(article) or data["country"]
    data["victim_name"] = clean_victim_name(data.get("victim_name"))
    data["incident_date"] = plausible_incident_date(article)
    return data


def incident_country_of(incident_countries, outlet_country: str) -> str:
    """Case-level incident country: the most common non-null article
    `incident_country` (ISO GB maps to the internal code UK); the outlet's
    country if none is set."""
    from collections import Counter
    votes = Counter("UK" if c == "GB" else c for c in incident_countries if c)
    return votes.most_common(1)[0][0] if votes else outlet_country


def refresh_case_country(db, case, Article) -> None:
    """Re-vote the case's country from its articles' incident_country values
    (a DE-outlet case about London becomes UK once any article says GB)."""
    votes = [icc for (icc,) in db.query(Article.incident_country)
             .filter(Article.case_id == case.id, Article.incident_country.isnot(None)).all()]
    if votes:
        case.country = incident_country_of(votes, case.country)


def update_case_from_article_v2(case, article) -> None:
    _legacy.update_case_from_article(case, article)
    if is_placeholder_name(case.victim_name):
        case.victim_name = clean_victim_name(getattr(article, "victim_name", None))
    if case.incident_date is None:
        case.incident_date = plausible_incident_date(article)


def link_article_to_case_v2(db, article, Case, Article) -> Optional[int]:
    """v2 entry point used by cases.link_article_to_case when CASE_LINK_V2 is on."""
    existing = find_matching_case_v2(db, article, Case, Article)
    if isinstance(existing, int):  # agent applied a decision
        return existing
    if existing is not None:
        update_case_from_article_v2(existing, article)
        try:
            article.case_id = existing.id
        except Exception:
            pass
        db.add(existing)
        db.flush()
        refresh_case_country(db, existing, Article)
        print(f"    🔗 Linked to Case #{existing.id} ({existing.article_count} articles, {existing.country})", flush=True)
        return existing.id
    new_case = Case(**create_case_from_article_v2(article))
    db.add(new_case)
    db.flush()
    try:
        article.case_id = new_case.id
    except Exception:
        pass
    print(f"    📌 New Case #{new_case.id} ({new_case.country} {new_case.incident_date})", flush=True)
    return new_case.id
