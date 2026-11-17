"""
match_patterns.py
-----------------
Keyword gate used to decide which fetched feed items are worth keeping.

This public version only does case-insensitive literal keyword matching against
the terms configured for each country (see countries.py / examples/). The
per-language regular-expression patterns used by the original system are not
part of this repository.
"""
from __future__ import annotations

from typing import Iterable


def matches_patterns(text: str, langs: Iterable[str] | None = None) -> bool:
    """No regex patterns in this version."""
    return False


def matches(text: str, keywords: Iterable[str], langs: Iterable[str] | None = None) -> bool:
    """True if any configured keyword/phrase appears in `text` (case-insensitive)."""
    t = (text or "").lower()
    return any(kw.lower() in t for kw in keywords)


def has_death_signal(text: str, langs: Iterable[str] | None = None) -> bool:
    """Used only for optional near-miss logging; disabled in this version."""
    return False
