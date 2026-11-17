"""preprocess.py — small text helpers (HTML stripping, snippet truncation)."""
import re

_HTML = re.compile(r"<[^>]+>")


def clean_html(text: str) -> str:
    return re.sub(_HTML, " ", text or "").strip()


def snippet(text: str, max_chars: int = 400) -> str:
    plain = clean_html(text)
    if len(plain) <= max_chars:
        return plain
    cut = plain[:max_chars].rsplit(" ", 1)[0]
    return cut + "…"
