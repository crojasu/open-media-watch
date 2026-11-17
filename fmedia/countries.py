"""
countries.py
------------
Source configuration and fetchers: Google-News-style RSS feeds, GDELT DOC API,
and the keyword gate.

Countries, search terms and feeds are NOT defined in code: they are loaded from
a JSON file (default: examples/sources.example.json, placeholder terms only).
Point OMW_SOURCES at your own file.

Each country config has: name, languages, rss_feeds, gdelt_domains,
national_domains, and optionally rss_splits. Search terms per country have
tiers: modern / historical / violence / neutral (+ geo_suffix).
"""
from __future__ import annotations

import json
import os
import re
import time
import requests
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Generator, Dict, Any
from urllib.parse import quote_plus, urlparse

from . import match_patterns
from .log_buffer import log as _raw_log

def _log(msg, level="info"):
    _raw_log(msg, source="scan", level=level)


#  near-miss investigation (see fetch_rss_country): cap on how many
# discarded national-feed items get a near-miss log line PER fetch_rss_country()
# CALL (was per-process: a long-lived server went silent after the first 20 ever).
NEAR_MISS_CAP_PER_RUN = 20

# Google News RSS returns at most this many items per query; hitting it means the
# query was truncated and older/less-ranked stories were never seen.
GNEWS_ITEM_CAP = 100

# Pause between the extra split-query requests of one country (rate-limit courtesy;
# the countries themselves are fetched in parallel by jobs.py).
SPLIT_QUERY_DELAY_SECONDS = 1.0

#  observability: per-country counters of the last fetch_rss_country()
# run, reset at the start of each run. See get_gate_stats()/gate_summary().
_gate_stats: Dict[str, Dict[str, int]] = {}


# ── Configuration (loaded from JSON) ──────────────────────────────────────────

_DEFAULT_SOURCES = Path(__file__).resolve().parent.parent / "examples" / "sources.example.json"


def _load_sources() -> dict:
    path = Path(os.getenv("OMW_SOURCES", str(_DEFAULT_SOURCES)))
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


_SOURCES = _load_sources()
SEARCH_TERMS: Dict[str, Dict] = _SOURCES["search_terms"]
COUNTRIES: Dict[str, Dict] = {
    code: {**cfg, "national_domains": tuple(cfg.get("national_domains", ))}
    for code, cfg in _SOURCES["countries"].items()
}


def _neutral_enabled(override: bool | None = None) -> bool:
    """ discovery flag — read live from env (not cached at import time, so
    tests can monkeypatch os.environ per-test without a module reload — same
    convention as fmedia/case_agent.py's _link_enabled()). `override` lets callers
    force it on/off explicitly (used by tests and by the GDELT/Guardian rotation
    helpers below, which need to compare the flag-off and flag-on query variants
    within the same call)."""
    if override is not None:
        return override
    return os.getenv("DISCOVERY_NEUTRAL_TERMS", "false").strip().lower() in ("1", "true", "yes", "on")

def get_search_queries(
    country_code: str, year: int | None = None, include_neutral: bool | None = None
) -> list[str]:
    """
    Return search queries for a country, era-aware, WITH geo suffix.
    Used by Google News RSS (backfill + daily) where geo-targeting matters.

    include_neutral: adds the "neutral" tier on top of whichever
    era-selected terms would otherwise apply. None (default) reads
    DISCOVERY_NEUTRAL_TERMS live; pass True/False to override (tests, and the
    rotation helpers below).
    """
    terms = SEARCH_TERMS.get(country_code, {})
    if not terms:
        return []

    modern = terms.get("modern", [])
    historical = terms.get("historical", [])
    violence = terms.get("violence", [])
    neutral = terms.get("neutral", [])
    suffix = terms.get("geo_suffix", "")

    if year is None:
        year = date.today().year

    if year < 2012:
        base = historical + violence
    elif year < 2018:
        base = historical + modern + violence
    else:
        base = modern + violence

    if _neutral_enabled(include_neutral):
        base = base + neutral

    if suffix:
        return [f"{q} {suffix}" for q in base]
    return base


def get_base_keywords(country_code: str, include_neutral: bool | None = None) -> list[str]:
    """
    Return flat keyword list (no geo suffix) for RSS text matching.
    Combines all eras — more inclusive for daily scanning.

    include_neutral: adds the "neutral" tier.
    None (default) reads the flag live; pass True/False to override.
    """
    terms = SEARCH_TERMS.get(country_code, {})
    if not terms:
        return []
    categories = ["modern", "historical", "violence"]
    if _neutral_enabled(include_neutral):
        categories.append("neutral")
    keywords = []
    for category in categories:
        keywords.extend(terms.get(category, []))
    return keywords


def _or_query_from_terms(keywords: list[str]) -> str:
    """Join a flat keyword list into a Guardian/GDELT-style OR query string,
    quoting multi-word phrases. Extracted from build_or_query() so the GDELT
    rotation helpers below can build an OR query from an arbitrary
    subset of terms (e.g. the neutral tier alone) without re-implementing the
    quoting rule."""
    parts = []
    for kw in keywords:
        if " " in kw:
            parts.append(f'"{kw}"')
        else:
            parts.append(kw)
    return " OR ".join(parts)


def build_or_query(country_code: str, include_neutral: bool | None = None) -> str:
    """
    Build an OR-based query string for APIs (Guardian, GDELT).
    Uses base keywords (no geo suffix) since these APIs have their own geo-filtering.

    include_neutral: adds the "neutral" tier.
    None (default) reads the flag live; pass True/False to override.
    """
    keywords = get_base_keywords(country_code, include_neutral=include_neutral)
    return _or_query_from_terms(keywords)


# Guardian's Content API `q` parameter has no published hard length limit, and
# unlike GDELT (fetched per-domain, see _gdelt_query_for_domain() above) there's
# no natural per-request axis to rotate the neutral tier across — Guardian is
# queried once per day (or once per backfill range). So instead of concatenating
# core+neutral without bound, or inventing a speculative day-parity rotation for
# a single yearly-called API, this falls back to the core (non-neutral) query for
# that call and logs a warning if the widened query would cross the safety
# margin below — "don't concatenate infinitely" without pretending there's a
# meaningful rotation axis where none exists.
GUARDIAN_QUERY_MAX_CHARS = 1800


def guardian_query(country_code: str = "UK", include_neutral: bool | None = None) -> str:
    """Build the Guardian API `q` query, widened with the neutral tier (,
    DISCOVERY_NEUTRAL_TERMS) when enabled. See GUARDIAN_QUERY_MAX_CHARS above for
    the fallback behaviour if the widened query gets too long."""
    core_query = build_or_query(country_code, include_neutral=False)
    if not _neutral_enabled(include_neutral):
        return core_query

    widened = build_or_query(country_code, include_neutral=True)
    if len(widened) <= GUARDIAN_QUERY_MAX_CHARS:
        return widened

    _log(
        f"[Guardian] Widened query ({len(widened)} chars) exceeds safety margin "
        f"({GUARDIAN_QUERY_MAX_CHARS}) — falling back to core query for this call.",
        level="error",
    )
    return core_query



ACTIVE_COUNTRIES = list(COUNTRIES.keys())  # All enabled by default


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_date(s: str) -> date:
    s = s.strip()
    for fmt in [
        "%a, %d %b %Y %H:%M:%S GMT",   # 'Fri, 25 Jul 2025 07:00:00 GMT'
        "%a, %d %b %Y %H:%M:%S %z",    # 'Fri, 25 Jul 2025 07:00:00 +0000'
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d",
        "%Y%m%dT%H%M%SZ",
        "%Y%m%d",
    ]:
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            continue
    # last resort: try just the date part
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except Exception:
        pass
    return date.today()


def _matches(text: str, keywords: list, langs: list[str] | None = None) -> bool:
    return match_patterns.matches(text, keywords, langs)


def strip_gnews_title_suffix(title: str, link: str, source_name: str) -> str:
    # Google's suffix spelling varies (" - Página|12", " - DIE ZEIT"), so for its
    # links cut the last " - " segment whatever it says; otherwise only the exact name.
    if "news.google.com" in link:
        head, sep, _ = title.rpartition(" - ")
        return head.strip() if sep and head.strip() else title
    for suffix in (f" - {source_name}", f" – {source_name}"):
        if title.endswith(suffix):
            return title[: -len(suffix)].strip()
    return title


def is_national_feed(url: str) -> bool:
    return "site:" not in url


def site_feed_outlets(feeds: Dict[str, str]) -> Dict[str, str]:
    """First domain label of each `site:` feed -> its feed name ("bbc" -> "BBC News")."""
    outlets = {}
    for name, url in feeds.items():
        m = re.search(r"site:([\w.-]+)", url)
        if m:
            outlets[m.group(1).split(".")[0].lower()] = name
    return outlets


def national_outlet(
    item: ET.Element, allowed_domains: tuple[str, ...], known_outlets: Dict[str, str],
) -> str | None:
    """Outlet name for an item from an un-scoped Google News feed, or None when
    its host is not in allowed_domains (country is taken from the feed, so a
    foreign outlet would otherwise be mislabelled). Outlets that also have a
    `site:` feed reuse that feed's name so ids and source aggregates match."""
    src = item.find("source")
    if src is None:
        return None
    host = (urlparse(src.get("url", "")).hostname or "").lower()
    if not any(host == d or host.endswith("." + d) for d in allowed_domains):
        return None
    for label in host.split("."):
        if label in known_outlets:
            return known_outlets[label]
    return (src.text or "").strip() or host


def _make_id(country: str, source: str, url: str) -> str:
    slug = source.lower().replace(" ", "-")
    return f"{country.lower()}-{slug}-{url[-80:]}"


# ── RSS fetcher (all countries) ───────────────────────────────────────────────

def split_queries_enabled() -> bool:
    """Kill switch for  split queries (DISCOVERY_SPLIT_QUERIES=false)."""
    return os.getenv("DISCOVERY_SPLIT_QUERIES", "true").strip().lower() not in ("0", "false", "no", "off")


def build_split_feeds(country_code: str) -> Dict[str, str]:
    """Unscoped Google News feeds that fan a country's broad daily query out so no
    single query hits the ~100-item cap: the lead term x each region, plus every
    other term on its own. Data comes from COUNTRIES[cc]["rss_splits"]; {} when the
    country has none. Pure: no network."""
    spec = (COUNTRIES.get(country_code) or {}).get("rss_splits")
    if not spec:
        return {}
    base = "https://news.google.com/rss/search"
    tail = f"when:7d&hl={spec['hl']}&gl={spec['gl']}&ceid={spec['gl']}:{spec['hl']}"

    def _url(query: str) -> str:
        return f"{base}?q={quote_plus(query)}+{tail}"

    def _term(t: str) -> str:
        return f'"{t}"' if " " in t else t

    terms, feeds = spec["terms"], {}
    for region in spec["regions"]:
        feeds[f"Google News {country_code} ({region})"] = _url(f"{_term(terms[0])} {region}")
    for term in terms:
        feeds[f"Google News {country_code} [{term}]"] = _url(_term(term))
    return feeds


def get_gate_stats(country_code: str) -> Dict[str, int]:
    """Counters of the last fetch_rss_country(country_code) run (copy; {} if none)."""
    return dict(_gate_stats.get(country_code, {}))


def gate_summary(country_code: str) -> str:
    stats = get_gate_stats(country_code)
    if not stats:
        return f"[gate/{country_code}] no run recorded"
    return f"[gate/{country_code}] " + " ".join(f"{k}={v}" for k, v in stats.items())


def fetch_rss_country(country_code: str) -> Generator[Dict[str, Any], None, None]:
    """Fetch RSS feeds for a specific country using shared search terms."""
    config = COUNTRIES.get(country_code)
    if not config:
        return

    # Use shared base keywords (richer than legacy rss_keywords)
    keywords = get_base_keywords(country_code) or config["rss_keywords"]
    _log(f"[RSS/{country_code}] Using {len(keywords)} keywords: {keywords[:5]}…")

    allowed_domains = config.get("national_domains", ())
    known_outlets = site_feed_outlets(config["rss_feeds"])
    feeds = dict(config["rss_feeds"])
    base_feed_count = len(feeds)
    if split_queries_enabled():
        feeds.update(build_split_feeds(country_code))
    stats = _gate_stats[country_code] = {
        "seen": 0, "matched": 0, "gate_discarded": 0, "foreign_dropped": 0,
        "near_miss_logged": 0,
    }
    near_miss_on = os.getenv("NEAR_MISS_LOG_ENABLED", "false").strip().lower() in ("1", "true", "yes")
    langs = config.get("languages")
    seen_links: set[str] = set()
    for feed_index, (source_name, url) in enumerate(feeds.items()):
        if feed_index >= base_feed_count:
            time.sleep(SPLIT_QUERY_DELAY_SECONDS)
        _log(f"[RSS/{country_code}] Fetching {source_name}…")
        try:
            r = requests.get(url, timeout=15, headers={
                "User-Agent": "OpenMediaWatch/0.1"
            })
            r.raise_for_status()
            root = ET.fromstring(r.content)
        except Exception as e:
            _log(f"[RSS/{country_code}] Error {source_name}: {e}", level="error")
            continue

        items = (root.findall(".//item") or
                 root.findall(".//{http://www.w3.org/2005/Atom}entry"))
        matched = 0
        national = is_national_feed(url)

        if len(items) >= GNEWS_ITEM_CAP:
            _log(f"[RSS/{country_code}] {source_name}: {len(items)} items = Google News cap, "
                 f"query likely truncated — split it further", level="warning")

        for item in items:
            def _g(tag):
                el = item.find(tag)
                return (el.text or "").strip() if el is not None else ""

            link  = _g("link") or _g("guid")
            if link in seen_links:   # same story from another split query
                continue
            seen_links.add(link)
            stats["seen"] += 1
            title = _g("title")
            desc  = _g("description") or _g("summary") or ""

            # Clean HTML entities and tags from Google News RSS
            import re, html
            desc  = re.sub(r'<[^>]+>', ' ', desc)
            desc  = html.unescape(desc).strip()
            desc  = re.sub(r'\s+', ' ', desc)
            title = html.unescape(title).strip()

            outlet = source_name
            if national:
                outlet = national_outlet(item, allowed_domains, known_outlets)
                if outlet is None:
                    stats["foreign_dropped"] += 1
                    continue

            pub   = _g("pubDate") or _g("published") or _g("updated")

            title = strip_gnews_title_suffix(title, link, outlet)
            # Remove trailing "  OutletName" from snippet (Google News artifact)
            desc = re.sub(r'\s{2,}' + re.escape(outlet) + r'\s*$', '', desc).strip()

            if not _matches(f"{title} {desc}", keywords, langs):
                # : every regex-gate discard is COUNTED (stats, always
                # on, free). The opt-in near-miss LOG
                # only samples national-feed discards that carry a death/violence
                # word — sports/politics discards are the gate working, not misses —
                # capped per run, so it is reviewable without noise.
                stats["gate_discarded"] += 1
                if (
                    national
                    and near_miss_on
                    and stats["near_miss_logged"] < NEAR_MISS_CAP_PER_RUN
                    and match_patterns.has_death_signal(f"{title} {desc}", langs)
                ):
                    stats["near_miss_logged"] += 1
                    _log(f"[near-miss/{country_code}] {outlet}: {title} || {desc[:120]}", level="info")
                continue

            matched += 1
            stats["matched"] += 1
            yield {
                "guardian_id":    _make_id(country_code, outlet, link),
                "title":          title,
                "web_url":        link,
                "thumbnail":      None,
                "section":        None,
                "published_date": _parse_date(pub) if pub else date.today(),
                "country":        country_code,
                "source":         outlet,
                "content_snippet": desc[:400],
                "_body":          f"{title}. {desc}",
            }

        _log(f"[RSS/{country_code}] {source_name}: {matched} matched of {len(items)} items")

    _log(gate_summary(country_code))


# ── GDELT fetcher (per country) ───────────────────────────────────────────────

GDELT_MAX_DAYS = 90

# Optional GDELT GKG theme clause (e.g. "theme:SOME_THEME") ORed into widened queries
# when DISCOVERY_NEUTRAL_TERMS is on. Set "gdelt_theme" in the sources JSON; empty = none.
GDELT_THEME = _SOURCES.get("gdelt_theme", "")

# GDELT's practical query-length behaviour is undocumented (no published hard
# limit found), but very long OR-chains are known in practice to degrade or
# trigger errors/429s — hence sources.py already fetches per-domain rather than
# one giant multi-domain query. Once core+neutral(+theme) crosses this safety
# margin we ROTATE tiers per domain (using the existing per-domain loop as the
# rotation axis) instead of concatenating without bound.
GDELT_QUERY_MAX_CHARS = 700


def _gdelt_query_for_domain(country_code: str, domain: str, index: int) -> str:
    """Build the GDELT query for one domain in the fetch_gdelt_country loop below.

    DISCOVERY_NEUTRAL_TERMS OFF (default): identical to the basic query —
    core keywords only, no theme clause.

    DISCOVERY_NEUTRAL_TERMS ON: widens with the neutral phrase tier and
    the theme clause. If the widened query fits under GDELT_QUERY_MAX_CHARS, every
    domain gets the full widened query (small country term lists — the common
    case). If not, domains alternate by index: even domains get the core query
    widened with the theme clause, odd domains get the neutral-only terms widened
    with the theme clause — every domain is still scanned every run, just with a
    rotating (not simultaneously concatenated) slice of the term set.
    """
    core_query = build_or_query(country_code, include_neutral=False)
    if not _neutral_enabled():
        return f"{core_query} domain:{domain}"

    full_query = build_or_query(country_code, include_neutral=True)
    full_widened = f"({full_query} OR {GDELT_THEME})" if GDELT_THEME else f"({full_query})"
    if len(full_widened) <= GDELT_QUERY_MAX_CHARS:
        return f"{full_widened} domain:{domain}"

    neutral_terms = SEARCH_TERMS.get(country_code, {}).get("neutral", [])
    if index % 2 == 0 or not neutral_terms:
        query = f"({core_query} OR {GDELT_THEME})" if GDELT_THEME else f"({core_query})"
    else:
        neutral_only = _or_query_from_terms(neutral_terms)
        query = f"({neutral_only} OR {GDELT_THEME})" if GDELT_THEME else f"({neutral_only})"
    return f"{query} domain:{domain}"


def fetch_gdelt_country(
    country_code: str,
    start_date: date,
    end_date: date,
) -> Generator[Dict[str, Any], None, None]:
    """Fetch GDELT articles for a specific country using shared search terms."""
    config = COUNTRIES.get(country_code)
    if not config:
        return

    cutoff = date.today() - timedelta(days=GDELT_MAX_DAYS)
    if end_date < cutoff:
        _log(f"[GDELT/{country_code}] Beyond 90-day limit — skipping.")
        return

    start_date = max(start_date, cutoff)
    domains    = config["gdelt_domains"]

    # Build query from shared terms (same as backfill) — logged for visibility;
    # the actual per-domain query (below) may rotate tiers, see
    # _gdelt_query_for_domain()'s docstring.
    or_query = build_or_query(country_code)
    _log(f"[GDELT/{country_code}] Query ({len(or_query)} chars): {or_query[:120]}…")

    for i, domain in enumerate(domains):
        days_back = max(1, min((date.today() - start_date).days, GDELT_MAX_DAYS))
        query = _gdelt_query_for_domain(country_code, domain, i)

        retries = 3
        data = {}
        for attempt in range(retries):
            try:
                r = requests.get(
                    "https://api.gdeltproject.org/api/v2/doc/doc",
                    params={
                        "query":      query,
                        "mode":       "artlist",
                        "maxrecords": 250,
                        "timespan":   f"{days_back}d",
                        "format":     "json",
                    },
                    timeout=30,
                )
                if r.status_code == 429:
                    wait = 10 * (attempt + 1)
                    _log(f"[GDELT/{country_code}] 429 on {domain} — waiting {wait}s…")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                data = r.json()
                break
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(5)
                else:
                    _log(f"[GDELT/{country_code}] Failed {domain}: {e}", level="error")
                    data = {}

        matched = 0
        for a in data.get("articles", []):
            try:
                seen = _parse_date(a.get("seendate", ""))
            except Exception:
                seen = date.today()

            if not (start_date <= seen <= end_date):
                continue

            url    = a.get("url", "")
            title  = a.get("title", "")
            matched += 1

            yield {
                "guardian_id":    _make_id(country_code, domain, url),
                "title":          title,
                "web_url":        url,
                "thumbnail":      None,
                "section":        None,
                "published_date": seen,
                "country":        country_code,
                "source":         domain,
                "content_snippet": title,
                "_body":          title,
            }

        _log(f"[GDELT/{country_code}] {domain}: {matched} articles in range")
        time.sleep(2)

