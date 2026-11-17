"""guardian.py — fetch articles from The Guardian API."""
from __future__ import annotations
import os, requests
from datetime import date
from typing import Generator, Dict, Any
from dotenv import load_dotenv
from .preprocess import snippet
from .log_buffer import log as _raw_log

load_dotenv()

def _log(msg, level="info"):
    _raw_log(msg, source="scan", level=level)

API_KEY = os.getenv("GUARDIAN_API_KEY", "test")

# Query built fresh per call (not frozen at import) so DISCOVERY_NEUTRAL_TERMS
# can be toggled without a code deploy — see countries.guardian_query().
from .countries import guardian_query


def fetch_day(day: date, max_pages: int = 10) -> Generator[Dict[str, Any], None, None]:
    """Yield raw article dicts for a single day."""
    query = guardian_query("UK")
    _log(f"[Guardian] Fetching {day} | query ({len(query)} chars): {query[:100]}…")
    yield from _fetch(str(day), str(day), max_pages, query)


def fetch_range(start: date, end: date, max_pages: int = 10) -> Generator[Dict[str, Any], None, None]:
    query = guardian_query("UK")
    _log(f"[Guardian] Fetching {start} → {end} | query ({len(query)} chars): {query[:100]}…")
    yield from _fetch(str(start), str(end), max_pages, query)


def _fetch(from_date: str, to_date: str, max_pages: int, query: str) -> Generator[Dict[str, Any], None, None]:
    page = 1
    total_yielded = 0
    while page <= max_pages:
        try:
            r = requests.get(
                "https://content.guardianapis.com/search",
                params=dict(
                    q           = query,
                    **{"from-date": from_date, "to-date": to_date},
                    **{"show-fields": "body,thumbnail"},
                    **{"page-size": 50},
                    page        = page,
                    **{"api-key": API_KEY},
                ),
                timeout=15,
            )
            r.raise_for_status()
            resp = r.json()["response"]
        except Exception as e:
            _log(f"[Guardian] Error page {page}: {e}", level="error")
            break

        for a in resp.get("results", []):
            fields = a.get("fields") or {}
            body   = fields.get("body", "")
            total_yielded += 1
            yield {
                "guardian_id":    a.get("id", ""),
                "title":          a.get("webTitle", ""),
                "web_url":        a.get("webUrl"),
                "thumbnail":      fields.get("thumbnail"),
                "section":        a.get("sectionName"),
                "published_date": a.get("webPublicationDate", "")[:10],
                "country":        "UK",
                "source":         "The Guardian",
                "content_snippet": snippet(body),
                "_body":          body,   # full body for ML, not stored
            }

        if page >= resp.get("pages", 1):
            break
        page += 1
    _log(f"[Guardian] {from_date}→{to_date}: {total_yielded} articles fetched across {page} page(s)")
