"""Full-text helpers for classification.

The classifier used to judge from title + ~100-char snippet only. These
helpers let it (a) read the stored article body when there is one, and
(b) fetch the body on demand before a rejection that matches the
configured recheck patterns becomes final.
"""
import ipaddress
import re
import socket
from typing import Optional
from urllib.parse import urlsplit

from .enrichment import resolve_gnews_url, scrape_article

# Text longer than this is never sent to the LLM (cost / prompt-size cap).
CLASSIFY_TEXT_CHARS = 1500
# A stored body shorter than this is treated as "no real text".
MIN_STORED_TEXT_CHARS = 100
SCRAPE_TIMEOUT_S = 10

from .taxonomy import RECHECK_REGEXES


def looks_like_violent_death(title: Optional[str], snippet: Optional[str]) -> bool:
    """True if the headline/snippet matches ALL configured `recheck_regexes`
    (taxonomy file). With none configured this is always False, i.e. no
    full-text re-check of rejections. Name kept for historical reasons."""
    if not RECHECK_REGEXES:
        return False
    text = f"{title or ''} {snippet or ''}"
    return all(r.search(text) for r in RECHECK_REGEXES)


def has_usable_full_text(art) -> bool:
    body = getattr(art, "full_text", None)
    return bool(body) and len(body) >= MIN_STORED_TEXT_CHARS


def is_safe_public_url(url: Optional[str]) -> bool:
    """http(s) only, and the host must not resolve to a private, loopback,
    link-local or otherwise non-public address (basic SSRF guard: article
    URLs come from third-party feeds). A host that does not resolve at all
    is harmless (nothing to reach) and left to fail in the scraper."""
    try:
        parts = urlsplit(url or "")
        host = parts.hostname
        if parts.scheme not in ("http", "https") or not host:
            return False
        port = parts.port or (443 if parts.scheme == "https" else 80)
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        return True
    except (ValueError, OSError):
        return False
    return all(
        ipaddress.ip_address(info[4][0].split("%")[0]).is_global for info in infos
    )


def text_for_classification(art) -> str:
    """Stored full text (capped) when present, else the short snippet."""
    body = getattr(art, "full_text", None)
    if body and len(body) >= MIN_STORED_TEXT_CHARS:
        return body[:CLASSIFY_TEXT_CHARS]
    return art.content_snippet or ""


def ensure_full_text(art) -> Optional[str]:
    """Return the article body, scraping and storing it if missing.

    Never raises: any scraper/resolver failure returns None so the caller can
    fall back to "not verifiable". Does not commit; the caller owns the
    transaction.
    """
    if has_usable_full_text(art):
        return art.full_text
    url = art.web_url
    if not is_safe_public_url(url):
        return None
    try:
        if urlsplit(url).hostname == "news.google.com":
            url = resolve_gnews_url(url)
            if not is_safe_public_url(url):
                return None
        text = scrape_article(url, timeout=SCRAPE_TIMEOUT_S)
    except Exception:
        return None
    if not text:
        return None
    art.full_text = text
    return text
