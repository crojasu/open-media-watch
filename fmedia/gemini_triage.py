"""
gemini_triage.py — Gemini Flash-Lite binary triage.
Free tier: 1,000 RPD, 15 RPM, 250K TPM.
Returns {"is_relevant": bool, "confidence": float} or None on error.
"""
import os, json, time, datetime
from typing import Optional
from .log_buffer import log as _log

def _triage_log(msg, level="info"):
    _log(msg, source="triage", level=level)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash-lite"

from .taxonomy import TRIAGE_PROMPT  # prompt text lives in the taxonomy file

# Lazy-initialized client
_client = None
# Circuit breaker: trips on daily quota or repeated rate limits
_daily_exhausted = False
_daily_exhausted_date: Optional[datetime.date] = None
_consecutive_rate_limits = 0

def _get_client():
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def _is_daily_limit(error_str: str) -> bool:
    """Check if the 429 is a daily quota (kill triage) vs per-minute (just wait)."""
    return "PerDay" in error_str or "per_day" in error_str


def _parse_retry_delay(error_str: str) -> float:
    """Extract retryDelay seconds from error, default 60s."""
    import re
    match = re.search(r"retryDelay.*?(\d+)", error_str)
    return float(match.group(1)) if match else 60.0


def triage_article(title: str, snippet: str, country: str = "UK") -> Optional[dict]:
    """Binary triage: is this article about the monitored topic?
    Returns {"is_relevant": bool, "confidence": float, "tokens_in": int, "tokens_out": int}
    or None on error (caller should fall through to DeepSeek/Haiku).

    Rate limit handling:
    - Per-minute 429: wait and retry once (15 RPM limit)
    - Per-day 429: circuit breaker ON, skip rest of session (1,000 RPD limit)
    """
    global _daily_exhausted, _daily_exhausted_date, _consecutive_rate_limits
    # Auto-reset circuit breaker on new calendar day
    if _daily_exhausted and _daily_exhausted_date != datetime.date.today():
        _daily_exhausted = False
        _daily_exhausted_date = None
        _consecutive_rate_limits = 0
        _triage_log("[triage] Circuit breaker auto-reset (new day)", level="info")
    if not GEMINI_API_KEY or _daily_exhausted:
        return None

    max_attempts = 2
    for attempt in range(max_attempts):
        try:
            from google import genai
            from google.genai import types

            client = _get_client()
            user_msg = f"Country: {country}\nTitle: {title}\nSnippet: {snippet[:500]}"

            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[f"{TRIAGE_PROMPT}\n\n{user_msg}"],
                config=types.GenerateContentConfig(
                    max_output_tokens=50,
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )

            text = response.text.strip()
            result = json.loads(text)

            # Extract token usage
            usage = response.usage_metadata
            result["tokens_in"]  = usage.prompt_token_count if usage else 0
            result["tokens_out"] = usage.candidates_token_count if usage else 0

            _consecutive_rate_limits = 0  # reset on success
            return result

        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                _consecutive_rate_limits += 1
                if _is_daily_limit(err) or _consecutive_rate_limits >= 3:
                    _daily_exhausted = True
                    _daily_exhausted_date = datetime.date.today()
                    _triage_log(f"[triage] Quota exhausted (consecutive={_consecutive_rate_limits}) — circuit breaker ON", level="error")
                    return None
                elif attempt < max_attempts - 1:
                    wait = min(_parse_retry_delay(err), 65)
                    _triage_log(f"[triage] Per-minute rate limit, waiting {wait:.0f}s...", level="info")
                    time.sleep(wait)
                    continue
                else:
                    _triage_log("[triage] Rate limit persists after retry, skipping this article", level="error")
                    return None
            else:
                _triage_log(f"[triage] Gemini error: {e}", level="error")
                return None
