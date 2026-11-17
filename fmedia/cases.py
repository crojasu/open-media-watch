"""
cases.py
--------
Case grouping model and deduplication logic for Open Media Watch.

Each "Case" represents ONE real-world relevant incident incident.
Multiple articles can reference the same case.

Deduplication strategy (layered):
  1. Exact URL match    → already handled by guardian_id unique constraint
  2. Title similarity   → SequenceMatcher ratio > 0.70 within ±14 days, same country
  3. Victim name match  → normalized name + same country + ±30 days
  4. Location + date    → same city + same date + same country (unnamed victims)

When a new article is classified as relevant incident, we try to match it to an existing Case.
If no match → create a new Case. If match → link article to existing Case.
"""
from __future__ import annotations
import os
import re
import unicodedata


def _parse_age(value) -> "int | None":
    """Convert LLM age output to int, handling strings like '30s', 'early 40s', etc."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (ValueError, TypeError):
        pass
    m = re.search(r'\d+', str(value))
    return int(m.group()) if m else None
from datetime import date, timedelta
from difflib import SequenceMatcher
from typing import Optional

from sqlalchemy import or_, and_

from .taxonomy import NEGATIVE_TYPE


# ── Helper: normalize text for comparison ─────────────────────────────────────

def _normalize(text: str) -> str:
    """Lowercase, strip accents, remove punctuation, collapse whitespace."""
    if not text:
        return ""
    # NFD decomposition → strip combining chars (accents)
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_name(name: str) -> str:
    """Normalize a person's name for fuzzy matching.
    Handles: accents, case, extra whitespace, common prefixes."""
    if not name:
        return ""
    name = _normalize(name)
    # Remove common title prefixes
    for prefix in ["la victima ", "la señora ", "doña ", "mrs ", "ms ", "miss "]:
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name.strip()


def _title_similarity(a: str, b: str) -> float:
    """Similarity ratio between two normalized titles."""
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def _name_match(name_a: str, name_b: str) -> bool:
    """Check if two victim names refer to the same person.
    Handles partial matches (e.g. 'María López' vs 'María López García')."""
    a = _normalize_name(name_a)
    b = _normalize_name(name_b)
    if not a or not b:
        return False
    # Exact match after normalization
    if a == b:
        return True
    # One name is contained in the other (partial name in headline vs full name in article)
    if a in b or b in a:
        return True
    # Token overlap: if ≥ 2 name parts match (first + last name)
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    common = tokens_a & tokens_b
    if len(common) >= 2:
        return True
    # High similarity (handles typos)
    if SequenceMatcher(None, a, b).ratio() > 0.85:
        return True
    return False


def _dates_close(d1: Optional[date], d2: Optional[date], days: int = 30) -> bool:
    """Check if two dates are within `days` of each other. None = always match."""
    if d1 is None or d2 is None:
        return True  # Can't disprove — assume possible match
    return abs((d1 - d2).days) <= days


def _parse_date(d) -> Optional[date]:
    """Parse a date from string (YYYY-MM-DD) or date object."""
    if d is None:
        return None
    if isinstance(d, date):
        return d
    try:
        return date.fromisoformat(str(d)[:10])
    except (ValueError, TypeError):
        return None


# ── SQLAlchemy Model ──────────────────────────────────────────────────────────
# Case model is defined in database.py — import it from there.
# from .database import Case


# ── Case matching logic ──────────────────────────────────────────────────────

def find_matching_case(db, article, Case) -> Optional[object]:
    """
    Try to find an existing Case that matches this article.
    
    Strategy (in order of reliability):
      1. Victim name match + same country + dates within 30 days
      2. Same location + same date + same country (for unnamed victims)
      3. Title similarity > 0.70 + same country + dates within 14 days
    
    Args:
        db:      SQLAlchemy session
        article: Article object with victim data populated
        Case:    The Case model class
    
    Returns:
        Matching Case object, or None if no match found.
    """
    country = getattr(article, "country", None)
    if not country:
        return None

    victim_name = getattr(article, "victim_name", None)
    incident_date = _parse_date(getattr(article, "incident_date_extracted", None)
                                or getattr(article, "incident_date", None))
    incident_location = getattr(article, "incident_location", None)
    title = getattr(article, "title", None)
    pub_date = _parse_date(getattr(article, "published_date", None))

    # Use incident_date if available, fallback to published_date
    ref_date = incident_date or pub_date

    # ── Strategy 1: Victim name match ─────────────────────────────────────
    if victim_name and _normalize_name(victim_name):
        # Query candidates: same country, with a name, within broad date window
        date_window = timedelta(days=60)
        q = db.query(Case).filter(
            Case.country == country,
            Case.victim_name.isnot(None),
        )
        if ref_date:
            q = q.filter(
                or_(
                    Case.incident_date.is_(None),
                    and_(
                        Case.incident_date >= ref_date - date_window,
                        Case.incident_date <= ref_date + date_window,
                    )
                )
            )
        candidates = q.all()
        for case in candidates:
            if _name_match(victim_name, case.victim_name):
                return case

    # ── Strategy 2: Location + date (unnamed victims) ─────────────────────
    if incident_location and incident_date:
        norm_loc = _normalize(incident_location)
        q = db.query(Case).filter(
            Case.country == country,
            Case.incident_date == incident_date,
            Case.incident_location.isnot(None),
        )
        candidates = q.all()
        for case in candidates:
            if _normalize(case.incident_location or "") == norm_loc:
                return case

    # ── Strategy 3: Title similarity (last resort) ────────────────────────
    if title and ref_date:
        from .database import Article as ArticleModel  # avoid circular at top level
        date_window = timedelta(days=14)
        q = db.query(Case).filter(
            Case.country == country,
        )
        if ref_date:
            q = q.filter(
                or_(
                    Case.incident_date.is_(None),
                    and_(
                        Case.incident_date >= ref_date - date_window,
                        Case.incident_date <= ref_date + date_window,
                    )
                )
            )
        # Check title against linked articles' titles
        existing_articles = db.query(ArticleModel).filter(
            ArticleModel.country == country,
            ArticleModel.case_id.isnot(None),
            ArticleModel.review_status == "confirmed",
        )
        if ref_date:
            existing_articles = existing_articles.filter(
                ArticleModel.published_date >= ref_date - date_window,
                ArticleModel.published_date <= ref_date + date_window,
            )
        for existing_art in existing_articles.limit(500).all():
            if _title_similarity(title, existing_art.title or "") > 0.70:
                return db.get(Case, existing_art.case_id)

    return None


def create_case_from_article(article) -> dict:
    """
    Build a Case dict from article fields.
    Use this to create a new Case when no match is found.
    """
    return {
        "victim_name":          getattr(article, "victim_name", None),
        "victim_age":           _parse_age(getattr(article, "victim_age", None)),
        "victim_nationality":   getattr(article, "victim_nationality", None),
        "country":              getattr(article, "country", "UK"),
        "incident_location":    getattr(article, "incident_location", None),
        "incident_region":      getattr(article, "incident_region", None),
        "incident_date":        _parse_date(getattr(article, "incident_date_extracted", None)),
        "event_type":        getattr(article, "event_type", None),
        "event_relation":    getattr(article, "event_relation", None),
        "event_motive":      getattr(article, "event_motive", None),
        "perpetrator_name":     getattr(article, "perpetrator_name", None),
        "perpetrator_age":      _parse_age(getattr(article, "perpetrator_age", None)),
        "perpetrator_status":   getattr(article, "perpetrator_status", None),
        "had_prior_complaints": getattr(article, "had_prior_complaints", None),
        "children_orphaned":    getattr(article, "children_orphaned", None),
        "weapon":               getattr(article, "weapon", None),
        "summary":              getattr(article, "event_summary", None),
        "article_count":        1,
        "first_reported_at":    getattr(article, "published_date", None),
    }


def update_case_from_article(case, article) -> None:
    """
    Enrich an existing Case with data from a new article.
    Only fills in fields that are currently null (doesn't overwrite).
    Increments article_count.
    """
    case.article_count = (case.article_count or 1) + 1

    # Fields to enrich (only fill nulls)
    enrich_fields = [
        ("victim_name", "victim_name"),
        ("victim_age", "victim_age"),
        ("victim_nationality", "victim_nationality"),
        ("incident_location", "incident_location"),
        ("incident_region", "incident_region"),
        ("perpetrator_name", "perpetrator_name"),
        ("perpetrator_age", "perpetrator_age"),
        ("perpetrator_status", "perpetrator_status"),
        ("had_prior_complaints", "had_prior_complaints"),
        ("children_orphaned", "children_orphaned"),
        ("weapon", "weapon"),
    ]
    _age_fields = {"victim_age", "perpetrator_age"}
    for case_field, art_field in enrich_fields:
        if getattr(case, case_field, None) is None:
            value = getattr(article, art_field, None)
            if value is not None:
                if case_field in _age_fields:
                    value = _parse_age(value)
                if value is not None:
                    setattr(case, case_field, value)

    # Update incident_date if case has none
    if case.incident_date is None:
        new_date = _parse_date(getattr(article, "incident_date_extracted", None))
        if new_date:
            case.incident_date = new_date

    # Update summary if current is shorter
    new_summary = getattr(article, "event_summary", None)
    if new_summary:
        if not case.summary or len(new_summary) > len(case.summary):
            case.summary = new_summary


def link_article_to_case(db, article, Case) -> Optional[int]:
    """
    Main entry point: after classifying an article as relevant incident,
    call this to link it to an existing Case or create a new one.

    Returns the case_id.
    """
    ftype = getattr(article, "event_type", None)
    if not ftype or ftype == NEGATIVE_TYPE:
        return None

    # ── Linker v2 — default ON; CASE_LINK_V2=false restores the
    # legacy path below byte for byte (kill switch).
    from .case_linker import v2_enabled
    if v2_enabled():
        from .case_linker import link_article_to_case_v2
        from .database import Article as _ArticleModel
        return link_article_to_case_v2(db, article, Case, _ArticleModel)

    # ── Foreign-incident guard ─────────
    # An article whose incident happened in a DIFFERENT country than the
    # monitored outlet's own `country` must never create or join a LOCAL Case
    # — Case rows (and find_matching_case above) are scoped by `country`, so a
    # foreign incident grouped in would silently mislabel a Case as local.
    # The article stays `confirmed` in `articles` (it's a real relevant incident, see
    # llm_classifier.py's persistence of `incident_country`), just ungrouped —
    # placed BEFORE find_matching_case/the agent block so this short-circuits
    # both the classic difflib path and AGENT_LINK_ENABLED below.
    from .database import to_iso_country

    incident_country = getattr(article, "incident_country", None)
    article_country = getattr(article, "country", None)
    if (
        incident_country
        and article_country
        and incident_country not in (article_country, to_iso_country(article_country))
    ):
        return None

    # Try to find a matching case
    existing_case = find_matching_case(db, article, Case)

    if existing_case:
        # Link to existing case and enrich it
        update_case_from_article(existing_case, article)
        try:
            article.case_id = existing_case.id
        except (AttributeError, Exception):
            pass
        db.add(existing_case)
        print(f"    🔗 Linked to Case #{existing_case.id} "
              f"({existing_case.article_count} articles)", flush=True)
        return existing_case.id
    else:
        # Create new case
        case_data = create_case_from_article(article)
        new_case = Case(**case_data)
        db.add(new_case)
        db.flush()  # Get the ID
        try:
            article.case_id = new_case.id
        except (AttributeError, Exception):
            pass
        print(f"    📌 New Case #{new_case.id} "
              f"({new_case.country} {new_case.incident_date})", flush=True)
        return new_case.id


# ── Batch: link all unlinked confirmed articles ──────────────────────────────

def link_all_unlinked(db, Case, Article, country: str = None) -> dict:
    """
    Find all confirmed relevant incident articles without a case_id and link them.
    Useful for backfilling cases after adding victim extraction.
    
    Usage:
        from fmedia.database import SessionLocal, Article, Case
        db = SessionLocal()
        result = link_all_unlinked(db, Case, Article, country="CL")
    """
    q = db.query(Article).filter(
        Article.review_status == "confirmed",
        Article.event_type.isnot(None),
        Article.event_type != NEGATIVE_TYPE,
    )
    # Legacy linker could only match on a name or a location; v2
    # also links nameless/placeless articles (identical titles, placeless
    # same-day rule) or gives each its own case, so it takes every article.
    from .case_linker import v2_enabled
    if not v2_enabled():
        q = q.filter(
            or_(
                Article.victim_name.isnot(None),
                Article.incident_location.isnot(None),
            )
        )
    try:
        q = q.filter(Article.case_id.is_(None))
    except Exception:
        pass  # case_id column may not exist yet

    if country:
        q = q.filter(Article.country == country.upper())

    q = q.order_by(Article.published_date.asc())  # oldest first → they create cases
    articles = q.all()

    print(f"\n[cases] Linking {len(articles)} unlinked articles to cases…\n", flush=True)
    stats = {"total": len(articles), "new_cases": 0, "linked": 0, "errors": 0}

    for i, art in enumerate(articles):
        try:
            case_id = link_article_to_case(db, art, Case)
            if case_id:
                # Check if it was new or existing
                case = db.get(Case, case_id)
                if case and case.article_count == 1:
                    stats["new_cases"] += 1
                else:
                    stats["linked"] += 1
            if (i + 1) % 50 == 0:
                db.commit()
                print(f"  [{i+1}/{len(articles)}] committed…", flush=True)
        except Exception as e:
            stats["errors"] += 1
            print(f"  [{i+1}] Error: {e}", flush=True)

    db.commit()
    print(f"\n[cases] Done. New cases: {stats['new_cases']}, "
          f"Linked to existing: {stats['linked']}, Errors: {stats['errors']}\n", flush=True)
    return stats
