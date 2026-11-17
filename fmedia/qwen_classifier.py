"""
qwen_classifier.py — Qwen2.5 (Alibaba Cloud Dashscope) fallback classifier.
Uses OpenAI-compatible API at dashscope.aliyuncs.com.

Env var: DASHSCOPE_API_KEY
Model:   qwen-plus  ($0.80/M in, $2.00/M out — comparable to Haiku, excellent Spanish)

Pricing vs chain position:
  Claude Haiku 4.5   → primary   ($0.80/M in, caching saves 90%)
  DeepSeek V3        → fallback1 ($0.28/M in, auto-cache)
  Qwen2.5 qwen-plus  → fallback2 ($0.80/M in, best Spanish among Chinese models)
  GPT-4o             → fallback3 ($2.50/M in)
"""
import os, json, time
from typing import Optional
from .log_buffer import log as _log
from .llm_classifier import SYSTEM_PROMPT, _build_user_prompt

def _qwen_log(msg, level="info"):
    _log(msg, source="qwen", level=level)

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-plus")
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def classify_qwen(title: str, snippet: str, country: str) -> Optional[dict]:
    """Full classification using Qwen2.5 via Alibaba Cloud Dashscope.
    Returns same dict format as _classify_claude: {type, relacion, motivo, confidence, ...}
    """
    if not DASHSCOPE_API_KEY:
        return None

    from openai import OpenAI
    client = OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url=QWEN_BASE_URL,
        timeout=30.0,
    )

    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model=QWEN_MODEL,
                max_tokens=500,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": _build_user_prompt(title, snippet, country)},
                ],
            )

            result = json.loads(resp.choices[0].message.content.strip())
            result["provider"] = f"qwen/{QWEN_MODEL}"

            usage = resp.usage
            result["tokens_in"]        = usage.prompt_tokens
            result["tokens_out"]       = usage.completion_tokens
            result["cache_read_tokens"]  = 0
            result["cache_write_tokens"] = 0

            return result

        except Exception as e:
            err = str(e).lower()
            if "quota" in err or "billing" in err or "credit" in err or "insufficient" in err:
                _qwen_log(f"[qwen] Quota/billing error: {e}", level="error")
                return None
            if attempt == 0:
                _qwen_log(f"[qwen] Error (attempt 1): {e} — retrying in 5s...", level="error")
                time.sleep(5)
                continue
            _qwen_log(f"[qwen] Error: {e}", level="error")
            return None
