"""
deepseek_classifier.py — DeepSeek V3 classifier for backfill.
Uses OpenAI-compatible API at api.deepseek.com.
Cache hit: $0.028/M tokens (auto-cache on repeated system prompt).
"""
import os, json, time
from typing import Optional
from .log_buffer import log as _log
from .llm_classifier import SYSTEM_PROMPT, _build_user_prompt

def _ds_log(msg, level="info"):
    _log(msg, source="deepseek", level=level)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")


def classify_deepseek(title: str, snippet: str, country: str) -> Optional[dict]:
    """Full classification using DeepSeek V3 (OpenAI-compatible API).
    Returns same dict format as _classify_claude: {type, relacion, motivo, confidence, ...}
    """
    if not DEEPSEEK_API_KEY:
        return None

    from openai import OpenAI
    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
        timeout=30.0,
    )

    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model="deepseek-chat",
                max_tokens=500,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _build_user_prompt(title, snippet, country)},
                ],
            )

            result = json.loads(resp.choices[0].message.content.strip())
            result["provider"] = "deepseek/v3"

            usage = resp.usage
            result["tokens_in"] = usage.prompt_tokens
            result["tokens_out"] = usage.completion_tokens

            cache_info = getattr(usage, "prompt_tokens_details", None)
            if cache_info:
                cached = getattr(cache_info, "cached_tokens", 0) or 0
                result["cache_read_tokens"] = cached
                result["cache_write_tokens"] = 0
            else:
                result["cache_read_tokens"] = 0
                result["cache_write_tokens"] = 0

            return result

        except Exception as e:
            if attempt == 0:
                _ds_log(f"[deepseek] Error (attempt 1): {e} — retrying in 5s...", level="error")
                time.sleep(5)
                continue
            _ds_log(f"[deepseek] Error: {e}", level="error")
            return None
