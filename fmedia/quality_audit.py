"""
quality_audit.py — Auditoría de calidad de clasificaciones automáticas.

Toma una muestra de artículos clasificados por LLM (auto-approve o llm-*)
y los re-clasifica con Claude para detectar discrepancias.

Si el porcentaje de error supera DRIFT_THRESHOLD (15%), genera alerta.
Costo típico: ~$0.05 por auditoría de 50 artículos.
"""
from __future__ import annotations
import time
from datetime import datetime
from typing import Optional
from sqlalchemy import func, or_

from .log_buffer import log as _raw_log
from .database import Article, POSITIVE_TYPE_IDS
from .taxonomy import NEGATIVE_TYPE

def _log(msg, level="info"):
    _raw_log(msg, source="audit", level=level)

DRIFT_THRESHOLD = 0.15   # >15% discrepancy rate → alerta
DEFAULT_SAMPLE  = 50


def _audit_classify(title: str, snippet: str, country: str) -> Optional[dict]:
    """Re-classify with Claude using an explicit double-check framing."""
    import os, json
    from .llm_classifier import SYSTEM_PROMPT, _build_user_prompt, CLAUDE_MODEL

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None

    import anthropic
    client = anthropic.Anthropic(api_key=api_key, timeout=30.0)

    try:
        msg = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=512,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": _build_user_prompt(title, snippet, country)}],
        )
        text = msg.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            # Truncated JSON — extract fields via regex fallback
            import re
            type_m  = re.search(r'"type"\s*:\s*"([^"]+)"', text)
            conf_m  = re.search(r'"confidence"\s*:\s*([0-9.]+)', text)
            if not type_m or not conf_m:
                raise
            result = {
                "type": type_m.group(1),
                "confidence": float(conf_m.group(1)),
                "reason": "truncated",
            }
        result["tokens_in"]  = msg.usage.input_tokens
        result["tokens_out"] = msg.usage.output_tokens
        return result
    except Exception as e:
        _log(f"[audit] Claude error: {e}", level="error")
        return None


def audit_sample(db, sample_size: int = DEFAULT_SAMPLE) -> dict:
    """
    Re-classify a random sample of auto-classified articles.

    Returns a report dict with:
      - sample_size, checked, discrepancies, drift_rate
      - cases: list of discrepant articles with original vs new classification
      - alert: True if drift_rate > DRIFT_THRESHOLD
    """
    _log(f"[audit] Starting quality audit (sample={sample_size})…")

    # Select random sample of auto-classified articles
    articles = (
        db.query(Article)
        .filter(
            Article.review_status == "confirmed",
            Article.event_type.isnot(None),
            Article.title.isnot(None),
            or_(
                Article.reviewed_by.like("llm-%"),
                Article.reviewed_by == "auto-approve",
                Article.reviewed_by.like("triage-%"),
            ),
        )
        .order_by(func.random())
        .limit(sample_size)
        .all()
    )

    total    = len(articles)
    checked  = 0
    discrepancies = []
    total_cost = 0.0

    for i, art in enumerate(articles):
        result = _audit_classify(
            art.title or "",
            art.content_snippet or "",
            art.country or "UK",
        )
        if result is None:
            continue

        checked += 1

        # Cost tracking
        tokens_in   = result.get("tokens_in", 0)
        tokens_out  = result.get("tokens_out", 0)
        cost = tokens_in * 0.80 / 1e6 + tokens_out * 4.00 / 1e6
        total_cost += cost

        original_type = art.event_type or NEGATIVE_TYPE
        new_type      = result.get("type", NEGATIVE_TYPE)
        new_conf      = float(result.get("confidence", 0.0))

        # Discrepancy: type changed AND new classification has reasonable confidence
        if original_type != new_type and new_conf >= 0.75:
            # Classify severity
            both_positive = (original_type in POSITIVE_TYPE_IDS) and (new_type in POSITIVE_TYPE_IDS)
            severity = "LOW" if both_positive else "HIGH"

            discrepancies.append({
                "article_id":    art.id,
                "title":         (art.title or "")[:100],
                "country":       art.country,
                "original_type": original_type,
                "new_type":      new_type,
                "new_conf":      round(new_conf, 2),
                "reviewed_by":   art.reviewed_by,
                "severity":      severity,
                "url":           art.web_url or "",
            })
            _log(
                f"  [{i+1}] ⚠ DISCREPANCY [{severity}] {original_type} → {new_type} "
                f"conf={new_conf:.2f} '{(art.title or '')[:50]}'",
                level="error" if severity == "HIGH" else "info",
            )
        else:
            _log(f"  [{i+1}] ✓ OK {original_type} '{(art.title or '')[:55]}'")

        time.sleep(0.3)

    drift_rate = len(discrepancies) / checked if checked > 0 else 0.0
    alert      = drift_rate > DRIFT_THRESHOLD

    high_severity = [d for d in discrepancies if d["severity"] == "HIGH"]

    report = {
        "sample_size":    total,
        "checked":        checked,
        "discrepancies":  len(discrepancies),
        "high_severity":  len(high_severity),
        "drift_rate":     round(drift_rate, 3),
        "drift_pct":      round(drift_rate * 100, 1),
        "alert":          alert,
        "cost_usd":       round(total_cost, 4),
        "cases":          discrepancies,
        "generated_at":   datetime.utcnow().isoformat(),
    }

    _log(f"{'━'*55}")
    _log(f"  AUDITORÍA COMPLETA", level="success" if not alert else "error")
    _log(f"{'━'*55}")
    _log(f"  Muestra:         {total}")
    _log(f"  Verificados:     {checked}")
    _log(f"  Discrepancias:   {len(discrepancies)} ({report['drift_pct']}%)")
    _log(f"  Alta severidad:  {len(high_severity)}")
    _log(f"  Costo:           ${total_cost:.4f} USD")
    if alert:
        _log(
            f"  ⚠ ALERTA: drift {report['drift_pct']}% supera umbral {DRIFT_THRESHOLD*100:.0f}%",
            level="error",
        )
    _log(f"{'━'*55}")

    return report


def flag_discrepant_articles(db, discrepancies: list) -> int:
    """Mark high-severity discrepant articles for human review."""
    flagged = 0
    for d in discrepancies:
        if d["severity"] != "HIGH":
            continue
        art = db.get(Article, d["article_id"])
        if art:
            art.review_status = "pending"
            art.reviewed_by   = f"audit-flagged (was: {art.reviewed_by})"
            db.add(art)
            flagged += 1
    db.commit()
    return flagged
