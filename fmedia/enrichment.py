"""
enrichment.py — Scrape full article text and re-extract victim data.

1. Resolve Google News redirect URLs to real URLs
2. Scrape full text from the real URL
3. Re-run DeepSeek extraction with full text for better victim data

Can be used from the daily job or as a standalone script.
"""
import re
import time
import html as html_mod
from datetime import date, datetime, timedelta
from typing import Optional, Dict, Any

from .log_buffer import log as _raw_log
from .database import Article, POSITIVE_TYPE_IDS

# : tras MAX intentos se deja de reintentar; entre intentos se espera
# BASE * 2^(intentos-1) días (1, 2, 4, 8...).
MAX_ENRICHMENT_ATTEMPTS = 5
ENRICHMENT_BACKOFF_BASE_DAYS = 1


def _is_due(attempts: int, last_attempt_at: Optional[datetime], now: datetime) -> bool:
    """True si el artículo puede reintentarse (nunca probado o ya pasó el backoff)."""
    if attempts <= 0 or last_attempt_at is None:
        return True
    wait = timedelta(days=ENRICHMENT_BACKOFF_BASE_DAYS * 2 ** (attempts - 1))
    return now - last_attempt_at >= wait


def _select_pending_ids(db, limit: int, now: datetime) -> list:
    """Ids a enriquecer: nunca probados primero, luego menos intentos; dentro
    de cada grupo id desc. Excluye agotados (>= MAX) y los que siguen en backoff."""
    from sqlalchemy import func
    attempts_col = func.coalesce(Article.enrichment_attempts, 0)
    rows = db.query(
        Article.id, attempts_col, Article.enrichment_last_attempt_at,
    ).filter(
        Article.event_type.in_(list(POSITIVE_TYPE_IDS)),
        Article.web_url.isnot(None),
        (Article.victim_name.is_(None)) | (Article.victim_age.is_(None)),
        attempts_col < MAX_ENRICHMENT_ATTEMPTS,
    ).order_by(attempts_col.asc(), Article.id.desc()).all()
    due = [r[0] for r in rows if _is_due(r[1], r[2], now)]
    return due[:limit]


def _log(msg, level="info"):
    _raw_log(msg, source="enrich", level=level)


def resolve_gnews_url(gnews_url: str) -> Optional[str]:
    """Resolve Google News redirect URL to real article URL."""
    try:
        from googlenewsdecoder import new_decoderv1
        result = new_decoderv1(gnews_url, interval=1)
        if result.get("status"):
            return result["decoded_url"]
    except Exception:
        pass
    return None


def scrape_article(url: str, timeout: int = 15) -> Optional[str]:
    """Scrape article text from URL. Returns cleaned text or None.

    `timeout` defaults to 15s (unchanged behavior for existing callers). The
    case-linking agent's `fetch_full_article` tool (fmedia/case_agent.py,
    ) passes timeout=10 explicitly.
    """
    import requests

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()

        raw_html = r.text

        # Remove scripts, styles, nav, header, footer
        for tag in ['script', 'style', 'nav', 'header', 'footer']:
            raw_html = re.sub(rf'<{tag}[^>]*>.*?</{tag}>', ' ', raw_html, flags=re.DOTALL)

        # Try to find article body
        article_match = re.search(r'<article[^>]*>(.*?)</article>', raw_html, flags=re.DOTALL)
        if article_match:
            raw_html = article_match.group(1)

        # Strip remaining tags
        text = re.sub(r'<[^>]+>', ' ', raw_html)
        text = re.sub(r'\s+', ' ', text).strip()
        text = html_mod.unescape(text)

        # Trim to reasonable size for LLM
        if len(text) > 3000:
            text = text[:3000]

        if len(text) < 50:
            return None

        return text

    except Exception:
        return None


def enrich_article(art: Article, db) -> Dict[str, Any]:
    """Enrich a single article with scraped full text + DeepSeek extraction.
    Returns dict with status and fields updated.
    """
    from .deepseek_classifier import classify_deepseek
    from .llm_classifier import _compute_cost, _persist_cost

    url = art.web_url
    if not url:
        return {"status": "skip", "reason": "no_url"}

    # Step 1: Resolve Google News URL
    if "news.google.com" in url:
        real_url = resolve_gnews_url(url)
        if not real_url:
            return {"status": "skip", "reason": "resolve_failed"}
    else:
        real_url = url

    # Step 2: Scrape full text
    full_text = scrape_article(real_url)
    if not full_text:
        return {"status": "skip", "reason": "scrape_failed"}

    # Persist it — this used to be thrown away after the single
    # re-extraction call below, leaving content_snippet (just the RSS
    # title+summary) as the only stored text for every downstream use:
    # case-linking's victim/date/location extraction, any future retraining,
    # manual review. Best-effort, never blocks re-extraction.
    try:
        art.full_text = full_text
    except Exception:
        pass

    # Step 3: Re-extract with full text (DeepSeek → Claude fallback)
    result = classify_deepseek(art.title or "", full_text, art.country or "UK")
    if result is None:
        # Claude fallback — always available
        from .llm_classifier import _classify_claude
        result = _classify_claude(art.title or "", full_text[:1500], art.country or "UK")
    if result is None:
        return {"status": "skip", "reason": "llm_error"}

    # Extract victim fields
    updated_fields = []
    _age_fields = {"victim_age", "perpetrator_age"}
    for field in ["victim_name", "victim_age", "victim_nationality",
                  "incident_location", "incident_region",
                  "perpetrator_name", "perpetrator_age", "perpetrator_status",
                  "had_prior_complaints", "children_orphaned", "weapon",
                  "event_summary"]:
        val = result.get(field)
        if val is not None:
            if field in _age_fields:
                from .cases import _parse_age
                val = _parse_age(val)
            if val is not None:
                try:
                    setattr(art, field, val)
                    updated_fields.append(field)
                except Exception:
                    pass

    raw_date = result.get("incident_date")
    if raw_date:
        try:
            art.incident_date_extracted = date.fromisoformat(str(raw_date)[:10])
            updated_fields.append("incident_date_extracted")
        except Exception:
            pass

    cost = _compute_cost(result)
    _persist_cost(art, result, cost)
    db.add(art)

    return {
        "status": "enriched",
        "fields": updated_fields,
        "victim_name": result.get("victim_name"),
        "victim_age": result.get("victim_age"),
        "cost": cost,
    }


def enrich_pending_articles(db, limit: int = 50, delay: float = 1.0) -> Dict[str, Any]:
    """Enrich confirmed relevant incident articles missing victim data.
    Returns summary dict.
    """
    ids = _select_pending_ids(db, limit, datetime.utcnow())
    by_id = {a.id: a for a in db.query(Article).filter(Article.id.in_(ids)).all()} if ids else {}
    articles = [by_id[i] for i in ids if i in by_id]

    total = len(articles)
    enriched = 0
    failed = 0
    total_cost = 0.0

    _log(f"[enrich] Starting enrichment of {total} articles (limit={limit})")

    for i, art in enumerate(articles):
        result = enrich_article(art, db)

        # Registrar el intento, sea cual sea el resultado.
        art.enrichment_attempts = (art.enrichment_attempts or 0) + 1
        art.enrichment_last_attempt_at = datetime.utcnow()
        art.enrichment_last_result = (
            "enriched" if result["status"] == "enriched" else result.get("reason", "skip")
        )[:40]
        db.add(art)

        if result["status"] == "enriched":
            enriched += 1
            total_cost += result.get("cost", 0)
            name = result.get("victim_name", "")
            age = result.get("victim_age", "")
            _log(f"[enrich] [{i+1}/{total}] OK name={name} age={age} {(art.title or '')[:50]}")
        else:
            failed += 1
            _log(f"[enrich] [{i+1}/{total}] SKIP {result['reason']}: {(art.title or '')[:50]}")

        if (i + 1) % 10 == 0:
            db.commit()

        time.sleep(delay)

    db.commit()

    summary = {
        "total": total,
        "enriched": enriched,
        "failed": failed,
        "cost_usd": round(total_cost, 4),
    }
    _log(f"[enrich] Done: {summary}", level="success")
    return summary
