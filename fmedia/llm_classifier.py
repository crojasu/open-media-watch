"""
llm_classifier.py
-----------------
LLM classification cascade: Claude (primary, prompt caching) -> DeepSeek -> Qwen -> GPT-4o.

The categories and the prompt are NOT defined here: they are loaded from the
taxonomy file (see taxonomy.py and examples/). Each article gets a type, a
secondary "relacion" and "motivo" label, a confidence, and optional extracted
fields (names, place, date). Low-confidence results are skipped for human review.

Flow:
  classify_pending_articles(db) -> reads all pending -> calls LLM -> saves decision
"""
from __future__ import annotations
import os, time, json, pathlib
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

from .log_buffer import log as _log

def _llm_log(msg, level="info"):
    _log(msg, source="llm", level=level)

load_dotenv()

# ── Reclassify-all epoch ────────────────────────────────────────────────────
# Persisted timestamp marking when the current "Reclassify All" round started.
# Articles with reviewed_at >= this epoch have already been processed in the
# current round and are skipped, no matter how many days pass between batches.
# Use /tmp (writable filesystem on most hosts).
_RECLASS_EPOCH_FILE = pathlib.Path("/tmp/.omw_reclassify_epoch")

def get_reclassify_epoch() -> Optional[datetime]:
    """Read the stored epoch (start of current reclassify-all round)."""
    try:
        ts = _RECLASS_EPOCH_FILE.read_text().strip()
        return datetime.fromisoformat(ts) if ts else None
    except (FileNotFoundError, ValueError):
        return None

def set_reclassify_epoch(dt: Optional[datetime] = None):
    """Set (or clear) the reclassify-all epoch."""
    try:
        if dt is None:
            _RECLASS_EPOCH_FILE.unlink(missing_ok=True)
        else:
            _RECLASS_EPOCH_FILE.write_text(dt.isoformat())
    except OSError as e:
        _llm_log(f"[llm] Warning: could not write epoch file: {e}", level="error")

def clear_reclassify_epoch():
    """Clear the epoch when a full round is complete."""
    set_reclassify_epoch(None)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")

# ── Cancellation flag ────────────────────────────────────────────────────────
import threading
_cancel_event = threading.Event()

def request_cancel():
    """Signal the running classification to stop after the current article."""
    _cancel_event.set()
    _llm_log("[llm] Cancellation requested — will stop after current article.", level="error")

def is_cancelled() -> bool:
    return _cancel_event.is_set()

def _reset_cancel():
    _cancel_event.clear()

CONFIDENCE_THRESHOLD = 0.82  # slightly lower to capture edge cases

# ── Taxonomy + prompt come from the taxonomy file (see taxonomy.py) ──
from .taxonomy import (  # noqa: E402
    NEGATIVE_TYPE, POSITIVE_TYPES as POSITIVE_TYPE_IDS, TYPE_LABELS as TYPE_LABELS_ES,
    TYPE_COLORS, TYPE_EMOJI, RELACIONES, RELACION_LABELS as RELACION_LABELS_ES,
    RELACION_DEFAULT, RELACION_NA, MOTIVOS, MOTIVO_LABELS as MOTIVO_LABELS_ES,
    MOTIVO_DEFAULT, MOTIVO_NA, CLASSIFIER_PROMPT as SYSTEM_PROMPT,
)

# Legacy alias
SUBTYPES = RELACIONES
SUBTYPE_LABELS_ES = RELACION_LABELS_ES


def _build_user_prompt(title: str, snippet: str, country: str) -> str:
    country_names = {
        "UK": "United Kingdom", "CL": "Chile", "AR": "Argentina",
        "MX": "México", "ES": "España",
    }
    country_name = country_names.get(country, country)
    # NOTE: this is the MONITORED OUTLET's
    # country, not necessarily where the incident happened — determine
    # incident_country independently from the article text (see SYSTEM_PROMPT
    # FIELD 4). Do not treat this line as an answer to "where did it happen".
    return f"""Media outlet monitored country: {country_name} (this is where the outlet is based — the incident described may have happened in a DIFFERENT country; determine incident_country independently from the article text)
Title: {title or '(no title)'}
Snippet: {snippet or '(no snippet)'}

Classify this article."""


CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")

def _classify_claude(title: str, snippet: str, country: str) -> Optional[dict]:
    if not ANTHROPIC_API_KEY:
        return None
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, timeout=30.0)
    for attempt in range(2):
        try:
            msg = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=500,
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
            # Strip markdown fences if present
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            result = json.loads(text.strip())
            result["provider"] = f"claude/{CLAUDE_MODEL}"
            result["tokens_in"]  = msg.usage.input_tokens
            result["tokens_out"] = msg.usage.output_tokens
            result["cache_write_tokens"] = getattr(msg.usage, "cache_creation_input_tokens", 0) or 0
            result["cache_read_tokens"]  = getattr(msg.usage, "cache_read_input_tokens", 0) or 0
            return result
        except Exception as e:
            err = str(e).lower()
            if "credit" in err or "billing" in err or "quota" in err or "insufficient" in err or "payment" in err:
                _llm_log(f"[llm] ⚠️  CREDIT EXHAUSTED — Claude API: {e}", level="error")
                _llm_log(f"[llm] ⚠️  Add credits at https://console.anthropic.com/settings/billing", level="error")
                return None
            if attempt == 0:
                _llm_log(f"[llm] Claude error (attempt 1): {e} — retrying in 5s...", level="error")
                time.sleep(5)
                continue
            _llm_log(f"[llm] Claude error ({CLAUDE_MODEL}): {e}", level="error")
            return None


def _classify_gpt4o(title: str, snippet: str, country: str) -> Optional[dict]:
    if not OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=500,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": _build_user_prompt(title, snippet, country)},
            ],
        )
        result = json.loads(resp.choices[0].message.content.strip())
        result["provider"] = "gpt4o"
        result["tokens_in"]  = resp.usage.prompt_tokens
        result["tokens_out"] = resp.usage.completion_tokens
        return result
    except Exception as e:
        _llm_log(f"[llm] GPT-4o error: {e}", level="error")
        return None


def classify_article(title: str, snippet: str, country: str = "UK") -> Optional[dict]:
    """Classify a single article. Returns {type, confidence, reason, provider} or None.

    Fallback chain:
      Claude Haiku 4.5 (primary, prompt caching)
        → DeepSeek V3  (auto-cache, cheap)
          → Qwen2.5    (best Spanish among Chinese models)
            → GPT-4o   (last resort)
    """
    result = _classify_claude(title, snippet, country)
    if result is None:
        _llm_log("[llm] Claude failed — trying DeepSeek V3…", level="error")
        try:
            from .deepseek_classifier import classify_deepseek
            result = classify_deepseek(title, snippet, country)
        except Exception as e:
            _llm_log(f"[llm] DeepSeek import error: {e}", level="error")
    if result is None:
        _llm_log("[llm] DeepSeek failed — trying Qwen2.5…", level="error")
        try:
            from .qwen_classifier import classify_qwen
            result = classify_qwen(title, snippet, country)
        except Exception as e:
            _llm_log(f"[llm] Qwen import error: {e}", level="error")
    if result is None:
        _llm_log("[llm] Qwen failed — trying GPT-4o…", level="error")
        result = _classify_gpt4o(title, snippet, country)
    if result is None:
        _llm_log("[llm] All providers failed.", level="error")
    return result


def _compute_cost(result: dict) -> float:
    """Compute actual USD cost from token counts in the LLM result dict.

    Anthropic usage fields:
      - input_tokens: non-cache input tokens (user prompt, etc.)
      - cache_creation_input_tokens: tokens written to cache (1.25x base)
      - cache_read_input_tokens: tokens read from cache (0.10x base)
    These three are SEPARATE — input_tokens does NOT include cache tokens.
    """
    provider       = result.get("provider", "")
    tokens_in      = result.get("tokens_in", 0) or 0      # non-cache input
    tokens_out     = result.get("tokens_out", 0) or 0
    cache_write    = result.get("cache_write_tokens", 0) or 0
    cache_read     = result.get("cache_read_tokens", 0) or 0

    if "haiku" in provider:
        base_in = 0.80 / 1e6
        cost = (tokens_in * base_in
                + cache_write * base_in * 1.25
                + cache_read  * base_in * 0.10
                + tokens_out * 4.00 / 1e6)
    elif "sonnet" in provider:
        base_in = 3.00 / 1e6
        cost = (tokens_in * base_in
                + cache_write * base_in * 1.25
                + cache_read  * base_in * 0.10
                + tokens_out * 15.0 / 1e6)
    elif "deepseek" in provider:
        # DeepSeek V3: cache hit $0.028/M, cache miss $0.28/M, output $0.42/M
        cost = (tokens_in * 0.28 / 1e6          # cache miss (worst case)
                - cache_read * 0.252 / 1e6       # refund for cached portion (0.28 - 0.028)
                + tokens_out * 0.42 / 1e6)
    elif "gpt4o" in provider:
        cost = tokens_in * 2.50 / 1e6 + tokens_out * 10.0 / 1e6
    else:
        base_in = 15.0 / 1e6
        cost = (tokens_in * base_in
                + cache_write * base_in * 1.25
                + cache_read  * base_in * 0.10
                + tokens_out * 75.0 / 1e6)
    return cost


def _persist_cost(art, result: dict, cost: float):
    """Save LLM cost metadata to article (best-effort).
    llm_tokens_in stores TOTAL input tokens (regular + cache_write + cache_read).
    """
    try:
        tokens_in    = result.get("tokens_in", 0) or 0
        cache_write  = result.get("cache_write_tokens", 0) or 0
        cache_read   = result.get("cache_read_tokens", 0) or 0
        art.llm_model_used = result.get("provider", "")
        art.llm_tokens_in  = tokens_in + cache_write + cache_read
        art.llm_tokens_out = result.get("tokens_out", 0) or 0
        art.llm_cost_usd   = cost
    except (AttributeError, Exception):
        pass  # columns may not exist yet


_TOKEN_KEYS = ("tokens_in", "tokens_out", "cache_write_tokens", "cache_read_tokens")


def _verify_rejection_with_full_text(art, first: dict, title: str,
                                     country_code: str, ensure_full_text) -> dict:
    """A violent-death article was rejected from a headline + short snippet.
    Fetch the body and re-judge; if the body cannot be fetched the rejection
    is unverifiable, so cap its confidence below the threshold and let a human
    decide (the article stays pending instead of being rejected for good)."""
    text = ensure_full_text(art)
    if not text:
        capped = dict(first)
        capped["confidence"] = min(float(first.get("confidence", 0.0)), 0.5)
        capped["_unverified"] = True
        return capped
    second = classify_article(title, text[:1500], country_code)
    if second is None:
        capped = dict(first)
        capped["confidence"] = min(float(first.get("confidence", 0.0)), 0.5)
        capped["_unverified"] = True
        return capped
    merged = dict(second)
    for key in _TOKEN_KEYS:  # both LLM calls are billed
        merged[key] = (first.get(key, 0) or 0) + (second.get(key, 0) or 0)
    return merged


def _classify_one(art, i: int, total: int, counts: dict, db, force_decide: bool = False) -> float:
    """
    Classify a single article, update counts and db object.
    Returns the actual cost incurred (in USD).
    If force_decide=True, always confirm/reject (never skip on low confidence).
    """
    # Pricing depends on model
    if "haiku" in CLAUDE_MODEL:
        CLAUDE_INPUT_PRICE  = 0.80 / 1_000_000
        CLAUDE_OUTPUT_PRICE = 4.00 / 1_000_000
    elif "sonnet" in CLAUDE_MODEL:
        CLAUDE_INPUT_PRICE  = 3.00 / 1_000_000
        CLAUDE_OUTPUT_PRICE = 15.0 / 1_000_000
    else:
        CLAUDE_INPUT_PRICE  = 15.0 / 1_000_000
        CLAUDE_OUTPUT_PRICE = 75.0 / 1_000_000
    INPUT_TOKENS        = 1200
    OUTPUT_TOKENS       = 180
    COST_PER_ARTICLE    = INPUT_TOKENS * CLAUDE_INPUT_PRICE + OUTPUT_TOKENS * CLAUDE_OUTPUT_PRICE

    EMOJI = TYPE_EMOJI

    from .fulltext import (
        ensure_full_text, has_usable_full_text, looks_like_violent_death,
        text_for_classification,
    )

    title        = art.title or ""
    # Stored full text (capped) when we have it, else the short snippet.
    snippet      = text_for_classification(art)
    country_code = art.country or "UK"
    violent_death = looks_like_violent_death(title, art.content_snippet or snippet)

    # ── Gemini triage: cheap/free binary filter before Haiku ──
    from .gemini_triage import triage_article
    triage = triage_article(title, snippet, country_code)

    if triage is not None:
        is_fem      = triage.get("is_relevant", True)  # default True = safe
        triage_conf = float(triage.get("confidence", 0.5))

        # HIGH confidence NOT relevant incident → reject immediately, skip Haiku
        # Never let the cheap triage drop an article that reports a death:
        # those always get the full classifier (and, if rejected, a text check).
        if not is_fem and triage_conf <= 0.25 and not violent_death:
            counts["rejected"] += 1
            art.review_status     = "rejected"
            art.human_label       = 0
            art.reviewed_by       = "triage-gemini"
            art.event_type     = NEGATIVE_TYPE
            art.event_relation = RELACION_NA
            art.event_motive   = MOTIVO_NA
            art.ml_score          = 1.0 - triage_conf
            art.llm_classified_at = datetime.utcnow()
            _persist_cost(art, {
                "provider": "gemini/triage",
                "tokens_in": triage.get("tokens_in", 0),
                "tokens_out": triage.get("tokens_out", 0),
            }, 0.0)
            db.add(art)
            _llm_log(f"  [{i+1}/{total}] ⚡ TRIAGE-SKIP conf={triage_conf:.2f} {title[:55]}")
            return 0.0

        # Otherwise fall through to Haiku for full typology classification
    # If triage is None (Gemini down/no key) → fall through to Haiku

    try:
        result = classify_article(title, snippet, country_code)
        if (result is not None and result.get("type") == NEGATIVE_TYPE
                and violent_death and not has_usable_full_text(art)):
            result = _verify_rejection_with_full_text(
                art, result, title, country_code, ensure_full_text)
        if result is None:
            counts["errors"] += 1
            _llm_log(f"  [{i+1}/{total}] ✗ ERROR — both APIs failed: {title[:55]}", level="error")
            return 0.0

        ftype      = result.get("type", NEGATIVE_TYPE)
        relacion   = result.get("relacion", result.get("subtype", RELACION_NA))
        motivo     = result.get("motivo", MOTIVO_DEFAULT)
        confidence = float(result.get("confidence", 0.0))
        provider   = result.get("provider", "llm")

        if ftype not in (POSITIVE_TYPE_IDS | {NEGATIVE_TYPE}):
            ftype = NEGATIVE_TYPE
        if relacion not in RELACIONES:
            relacion = RELACION_DEFAULT
        if motivo not in MOTIVOS:
            motivo = MOTIVO_DEFAULT

        real_cost = _compute_cost(result)

        # An unverifiable rejection is never final, not even with force_decide.
        if (confidence < CONFIDENCE_THRESHOLD and not force_decide) or result.get("_unverified"):
            counts["skipped"] += 1
            # Save typology data even on SKIP so we don't re-process this article
            art.event_type     = ftype
            art.event_relation = relacion
            art.event_motive   = motivo
            art.ml_score          = confidence
            art.reviewed_at       = datetime.utcnow()
            try:
                art.llm_classified_at = datetime.utcnow()
            except Exception:
                pass
            _persist_cost(art, result, real_cost)
            db.add(art)
            label = TYPE_LABELS_ES.get(ftype, ftype)
            _llm_log(f"  [{i+1}/{total}] ⏭  SKIP conf={confidence:.2f} [{label}] {title[:55]}")
            return real_cost

        is_positive       = ftype in POSITIVE_TYPE_IDS
        art.review_status = "confirmed" if is_positive else "rejected"
        art.human_label   = 1 if is_positive else 0
        art.reviewed_by   = f"llm-{provider}"
        art.reviewed_at   = datetime.utcnow()
        try:
            art.llm_classified_at = datetime.utcnow()
        except Exception:
            pass  # column may not exist yet

        art.event_type     = ftype
        art.event_relation = relacion
        art.event_motive   = motivo

        # ── Save victim/case extraction data ──────────────────────────
        victim_fields = {
            "victim_name":          result.get("victim_name"),
            "victim_age":           result.get("victim_age"),
            "victim_nationality":   result.get("victim_nationality"),
            "incident_location":    result.get("incident_location"),
            "incident_region":      result.get("incident_region"),
            "perpetrator_name":     result.get("perpetrator_name"),
            "perpetrator_age":      result.get("perpetrator_age"),
            "perpetrator_status":   result.get("perpetrator_status"),
            "had_prior_complaints": result.get("had_prior_complaints"),
            "children_orphaned":    result.get("children_orphaned"),
            "weapon":               result.get("weapon"),
            "event_summary":     result.get("event_summary"),
        }
        _age_fields = {"victim_age", "perpetrator_age"}
        for field_name, value in victim_fields.items():
            if value is not None:
                if field_name in _age_fields:
                    from .cases import _parse_age
                    value = _parse_age(value)
                if value is not None:
                    try:
                        setattr(art, field_name, value)
                    except (AttributeError, Exception):
                        pass  # Column not yet in DB — safe to skip

        # incident_date → incident_date_extracted (Date column)
        raw_date = result.get("incident_date")
        if raw_date:
            try:
                from datetime import date as _date
                art.incident_date_extracted = _date.fromisoformat(str(raw_date)[:10])
            except (ValueError, AttributeError, Exception):
                pass

        # ── Incident country ──────────
        # Where the incident ACTUALLY happened, which may differ from
        # `art.country` (the monitored outlet's country). A mismatch here
        # means this article is a REAL, still-`confirmed` relevant incident, but
        # local_incident_filter() (fmedia/database.py) excludes it from every
        # country-scoped view, and cases.py's foreign-incident guard keeps it
        # out of local Case grouping. Only a well-formed ISO-2 code is stored;
        # anything else (missing, wrong length, non-alpha) is left as None,
        # same as "not determined".
        raw_incident_country = result.get("incident_country")
        if isinstance(raw_incident_country, str):
            candidate = raw_incident_country.strip().upper()
            if len(candidate) == 2 and candidate.isalpha():
                try:
                    art.incident_country = candidate
                except (AttributeError, Exception):
                    pass  # column may not exist yet

        _persist_cost(art, result, real_cost)
        db.add(art)

        if is_positive:
            # — embed_article() was only ever wired into
            # the human-review confirm endpoint and the 2h-later auto-approve
            # sweep (jobs.py), never into this function — which is the path
            # EVERY normal daily classification and backlog/force_decide pass
            # actually confirms articles through. Result: only 1,037 of
            # 23,169 confirmed articles (4.5%) ever got a vector, so the
            # case-linking agent's semantic search had almost nothing to
            # search. Embed here too, before case-linking, so every new
            # confirmation keeps the vector index current going forward
            # instead of falling further behind a backfill script.
            try:
                from .embeddings import embed_article
                embed_article(db, art)
            except Exception as e:
                _llm_log(f"    [embed] embed_article failed (non-fatal): {e}", level="error")

        # ── Link article to Case (dedup/grouping) ────────────────
        if is_positive:
            try:
                from .cases import link_article_to_case
                from .database import Case
                link_article_to_case(db, art, Case)
            except Exception as e:
                _llm_log(f"    [case] link error (non-fatal): {e}", level="error")

        label     = TYPE_LABELS_ES.get(ftype, ftype)
        rel_label = RELACION_LABELS_ES.get(relacion, relacion)
        mot_label = MOTIVO_LABELS_ES.get(motivo, motivo)

        if is_positive:
            counts[ftype] = counts.get(ftype, 0) + 1
            emoji = EMOJI.get(ftype, "✓")
            _llm_log(f"  [{i+1}/{total}] {emoji} {label} / {rel_label} / {mot_label} conf={confidence:.2f} [{provider}] {title[:45]}", level="success")
        else:
            counts["rejected"] += 1
            _llm_log(f"  [{i+1}/{total}] ✗  Not relevant conf={confidence:.2f} [{provider}] {title[:55]}")

        return real_cost

    except Exception as e:
        counts["errors"] += 1
        _llm_log(f"  [{i+1}/{total}] ✗ ERROR: {e} — {title[:45]}", level="error")
        return 0.0


def _print_batch_summary(batch_num: int, batch_cost: float, total_cost: float,
                          batch_counts: dict, grand_counts: dict, batch_size: int) -> None:
    """Print a formatted summary after each batch of 100."""
    confirmed = sum(batch_counts.get(t, 0) for t in POSITIVE_TYPE_IDS)
    rejected  = batch_counts.get("rejected", 0)
    skipped   = batch_counts.get("skipped", 0)
    errors    = batch_counts.get("errors", 0)
    processed = confirmed + rejected + skipped + errors

    g_confirmed = sum(grand_counts.get(t, 0) for t in POSITIVE_TYPE_IDS)
    g_rejected  = grand_counts.get("rejected", 0)
    g_total     = g_confirmed + g_rejected

    rate = (g_confirmed / g_total * 100) if g_total > 0 else 0

    _llm_log(f"{'━'*60}")
    _llm_log(f"  BATCH {batch_num} COMPLETE — {processed} artículos procesados", level="success")
    _llm_log(f"{'━'*60}")
    _llm_log(f"  Este batch:")
    _llm_log(f"    ✓ Confirmed:  {confirmed:>4}  |  ✗ Rejected: {rejected:>4}  |  ⏭  Skip: {skipped}  |  ⚠ Errores: {errors}")
    # Type breakdown for this batch
    for ftype in POSITIVE_TYPE_IDS:
        n = batch_counts.get(ftype, 0)
        if n > 0:
            label = TYPE_LABELS_ES.get(ftype, ftype)
            _llm_log(f"    • {label}: {n}")
    _llm_log(f"  Costo este batch:   ${batch_cost:.4f} USD")
    _llm_log(f"{'─'*60}")
    _llm_log(f"  ACUMULADO:")
    _llm_log(f"    Confirmed: {g_confirmed:>5}  ({rate:.1f}% of all classified)")
    _llm_log(f"    Total cost so far: ${total_cost:.4f} USD  (~${total_cost/max(g_total,1)*1000:.2f}/1k artículos)")
    _llm_log(f"{'━'*60}")


def classify_articles_by_ids(article_ids: list[int]):
    """Classify specific articles by ID. Used for day-scoped classification from the daily fetch page."""
    from .database import SessionLocal, Article
    _reset_cancel()
    db = SessionLocal()
    try:
        articles = db.query(Article).filter(Article.id.in_(article_ids)).all()
        if not articles:
            _llm_log("[llm] No articles found for given IDs.")
            return {"total": 0}
        total = len(articles)
        _llm_log(f"[llm] Classifying {total} articles by ID (force_decide=True)…")
        counts = {t: 0 for t in list(POSITIVE_TYPE_IDS) + ["rejected", "skipped", "errors"]}
        for i, art in enumerate(articles):
            if is_cancelled():
                _llm_log("[llm] CANCELLED by user.", level="error")
                db.commit()
                break
            _classify_one(art, i, total, counts, db, force_decide=True)
            if (i + 1) % 10 == 0:
                db.commit()
            time.sleep(0.3)
        db.commit()
        confirmed = sum(counts.get(t, 0) for t in POSITIVE_TYPE_IDS)
        _llm_log(f"[llm] Day classification complete: {total} processed, {confirmed} confirmed.", level="success")
        return {"total": total, "confirmed": confirmed, "rejected": counts.get("rejected", 0)}
    except Exception as e:
        _llm_log(f"[llm] classify_articles_by_ids error: {e}", level="error")
        return {"error": str(e)}
    finally:
        db.close()


def classify_pending_articles(
    db,
    country: Optional[str] = None,
    batch_size: int = 100,
    auto_confirm: bool = False,
    include_reviewed: bool = False,
    force_all: bool = False,
) -> dict:
    """
    Classify pending articles using LLM with UN typology.
    Processes in sub-batches of 100, showing cost + results after each one.

    Before starting, prints cost estimate and asks for confirmation (unless
    auto_confirm=True, which is used when called programmatically from jobs.py).

    Args:
        db:               SQLAlchemy session
        country:          Filter by country code (UK / CL / AR / MX / ES), or None for all
        batch_size:       Max total articles to process in this run (default 100)
        auto_confirm:     Skip interactive cost confirmation prompt (for scheduler/API use)
        include_reviewed: Also reclassify confirmed/rejected articles that are missing
                          event_type (typology not yet assigned by LLM)
        force_all:        Reclassify ALL articles regardless of current status/typology
    """
    from .database import Article

    SUBBATCH = 100   # articles per progress checkpoint
    # Pricing depends on model — Haiku is ~25x cheaper than Opus
    if "haiku" in CLAUDE_MODEL:
        CLAUDE_INPUT_PRICE  = 0.80 / 1_000_000
        CLAUDE_OUTPUT_PRICE = 4.00 / 1_000_000
    elif "sonnet" in CLAUDE_MODEL:
        CLAUDE_INPUT_PRICE  = 3.00 / 1_000_000
        CLAUDE_OUTPUT_PRICE = 15.0 / 1_000_000
    else:  # opus
        CLAUDE_INPUT_PRICE  = 15.0 / 1_000_000
        CLAUDE_OUTPUT_PRICE = 75.0 / 1_000_000
    COST_PER_ARTICLE    = 1200 * CLAUDE_INPUT_PRICE + 180 * CLAUDE_OUTPUT_PRICE

    # ── 1. Load articles ─────────────────────────────────────────────────────
    if force_all:
        # Reclassify ALL articles (e.g. after prompt changes).
        # Uses a persisted "epoch" timestamp: on the FIRST call, we store
        # now() as the epoch.  We then only pick articles whose reviewed_at
        # is BEFORE the epoch (or NULL) — i.e. not yet touched in this round.
        # This lets you run batches over days without re-processing.
        from sqlalchemy import or_
        epoch = get_reclassify_epoch()
        if epoch is None:
            # First batch of a new round — set the epoch
            epoch = datetime.utcnow()
            set_reclassify_epoch(epoch)
            _llm_log(f"[llm] New reclassify-all round started at {epoch.isoformat()}")
        else:
            _llm_log(f"[llm] Continuing reclassify-all round from {epoch.isoformat()}")
        q = db.query(Article).filter(
            Article.title.isnot(None),
            or_(Article.reviewed_at == None, Article.reviewed_at < epoch),  # noqa: E711
        )
    elif include_reviewed:
        # Reclassify: articles missing typology (regardless of review_status)
        q = db.query(Article).filter(Article.event_type == None)  # noqa: E711
        q = q.order_by(Article.fetched_at.desc())
    else:
        # Only pending articles that haven't been LLM-classified yet.
        # Newest first: a large historical backlog (e.g. from a bulk import)
        # must never starve today's freshly-fetched articles of classification.
        q = db.query(Article).filter(
            Article.review_status == "pending",
            Article.event_type == None,  # noqa: E711
        )
        q = q.order_by(Article.fetched_at.desc())
    if country:
        cc_list = [c.strip().upper() for c in country.split(",") if c.strip()]
        q = q.filter(Article.country.in_(cc_list)) if len(cc_list) > 1 else q.filter(Article.country == cc_list[0])
    articles = q.limit(batch_size).all()

    mode_label = "TODOS (force)" if force_all else ("reclasificar (sin tipología)" if include_reviewed else "pendientes")
    if not articles:
        if force_all and get_reclassify_epoch():
            clear_reclassify_epoch()
            _llm_log(f"[llm] Reclassify-all round COMPLETE — all articles processed. Epoch cleared.")
        _llm_log(f"[llm] No hay artículos {mode_label} para clasificar.")
        return {"total": 0, "confirmed": 0, "rejected": 0, "skipped": 0, "errors": 0}

    total = len(articles)
    estimated_cost = total * COST_PER_ARTICLE

    # ── 2. Cost estimate ──────────────────────────────────────────────────────
    _llm_log(f"{'━'*60}")
    _llm_log(f"  ESTIMACION DE COSTO — Clasificacion LLM")
    _llm_log(f"{'━'*60}")
    _llm_log(f"  Pais:              {country or 'todos'}")
    _llm_log(f"  Articulos:         {total:,}")
    _llm_log(f"  Sub-batches:       {(total + SUBBATCH - 1) // SUBBATCH} x {SUBBATCH}")
    _llm_log(f"  Modelo:            {CLAUDE_MODEL}")
    _llm_log(f"  Costo estimado:    ${estimated_cost:.4f} USD  (~${COST_PER_ARTICLE:.4f}/articulo)")
    _llm_log(f"{'━'*60}")

    # ── 3. Confirmation prompt (interactive only) ─────────────────────────────
    if not auto_confirm:
        try:
            answer = input(f"\n  ¿Continuar? [s/N]: ").strip().lower()
            if answer not in ("s", "si", "sí", "y", "yes"):
                print("  Cancelado.", flush=True)
                return {"total": 0, "confirmed": 0, "rejected": 0, "skipped": 0, "errors": 0}
        except (EOFError, KeyboardInterrupt):
            # Non-interactive context (called from scheduler/API) — proceed
            pass

    _reset_cancel()
    _llm_log(f"[llm] Iniciando clasificacion de {total} articulos en sub-batches de {SUBBATCH}…")

    # ── 4. Classify in sub-batches of SUBBATCH ────────────────────────────────
    grand_counts = {t: 0 for t in list(POSITIVE_TYPE_IDS) + ["rejected", "skipped", "errors"]}
    total_cost   = 0.0
    batch_num    = 0
    cancelled    = False

    for batch_start in range(0, total, SUBBATCH):
        if is_cancelled():
            cancelled = True
            _llm_log(f"[llm] CANCELLED by user after {batch_start} articles.", level="error")
            break

        batch_end   = min(batch_start + SUBBATCH, total)
        batch_arts  = articles[batch_start:batch_end]
        batch_num  += 1
        batch_counts = {t: 0 for t in list(POSITIVE_TYPE_IDS) + ["rejected", "skipped", "errors"]}
        batch_cost   = 0.0

        _llm_log(f"── Sub-batch {batch_num} ({batch_start+1}–{batch_end} / {total}) ──────────────────────────")

        for j, art in enumerate(batch_arts):
            if is_cancelled():
                cancelled = True
                _llm_log(f"[llm] CANCELLED by user at article {batch_start+j+1}/{total}.", level="error")
                db.commit()
                break

            i = batch_start + j
            cost = _classify_one(art, i, total, batch_counts, db)
            batch_cost  += cost
            total_cost  += cost

            if (j + 1) % 10 == 0:
                db.commit()

            time.sleep(0.3)

        db.commit()

        if cancelled:
            break

        # Update grand counts from batch
        for k in batch_counts:
            grand_counts[k] = grand_counts.get(k, 0) + batch_counts.get(k, 0)

        _print_batch_summary(batch_num, batch_cost, total_cost,
                              batch_counts, grand_counts, len(batch_arts))

    _reset_cancel()

    # ── 5. Final summary ──────────────────────────────────────────────────────
    total_confirmed = sum(grand_counts.get(t, 0) for t in POSITIVE_TYPE_IDS)
    grand_total_classified = total_confirmed + grand_counts.get("rejected", 0)
    precision = total_confirmed / grand_total_classified if grand_total_classified > 0 else 0

    _llm_log(f"{'═'*60}")
    _llm_log(f"  CLASSIFICATION COMPLETE", level="success")
    _llm_log(f"{'═'*60}")
    _llm_log(f"  Total processed:   {total}")
    _llm_log(f"  Confirmed:          {total_confirmed}  ({precision*100:.1f}% precision)")
    for ftype in POSITIVE_TYPE_IDS:
        n = grand_counts.get(ftype, 0)
        if n > 0:
            label = TYPE_LABELS_ES.get(ftype, ftype)
            _llm_log(f"    • {label}: {n}")
    _llm_log(f"  Rejected:           {grand_counts.get('rejected', 0)}")
    _llm_log(f"  Low confidence:     {grand_counts.get('skipped', 0)}")
    _llm_log(f"  Errors:             {grand_counts.get('errors', 0)}")
    _llm_log(f"  Total cost:   ${total_cost:.4f} USD")
    _llm_log(f"{'═'*60}")

    return {
        "total":               total,
        "confirmed":           total_confirmed,
        "rejected":            grand_counts.get("rejected", 0),
        "skipped":             grand_counts.get("skipped", 0),
        "errors":              grand_counts.get("errors", 0),
        "cost_usd":            round(total_cost, 4),
        **{t: grand_counts.get(t, 0) for t in POSITIVE_TYPE_IDS},
    }


# ── Cost estimation / dry run ─────────────────────────────────────────────────
def estimate_classification_cost(db, country: str = None, batch_size: int = None) -> dict:
    """
    Estimate LLM cost BEFORE running classify_pending_articles.
    Does NOT call any API — just counts pending articles and calculates cost.
    
    Usage:
        from fmedia.llm_classifier import estimate_classification_cost
        from fmedia.database import SessionLocal
        db = SessionLocal()
        result = estimate_classification_cost(db, country="CL")
        print(result)
    """
    from sqlalchemy import func

    # Claude Haiku 4.5 pricing (per token)
    CLAUDE_INPUT_PRICE  = 0.80 / 1_000_000  # $0.80/M input tokens
    CLAUDE_OUTPUT_PRICE = 4.00 / 1_000_000  # $4.00/M output tokens
    INPUT_TOKENS_PER_ARTICLE  = 1200       # system prompt + title + snippet
    OUTPUT_TOKENS_PER_ARTICLE = 180        # JSON response with victim data

    COST_PER_ARTICLE = (
        INPUT_TOKENS_PER_ARTICLE  * CLAUDE_INPUT_PRICE +
        OUTPUT_TOKENS_PER_ARTICLE * CLAUDE_OUTPUT_PRICE
    )

    query = db.query(func.count(Article.id)).filter(
        Article.review_status == "pending",
        Article.title.isnot(None),
    )
    if country:
        cc_list = [c.strip().upper() for c in country.split(",") if c.strip()]
        query = query.filter(Article.country.in_(cc_list)) if len(cc_list) > 1 else query.filter(Article.country == cc_list[0])
    if batch_size:
        total = min(query.scalar() or 0, batch_size)
    else:
        total = query.scalar() or 0

    total_cost = total * COST_PER_ARTICLE

    # Per-country breakdown
    breakdown = {}
    cc_list_for_breakdown = [c.strip().upper() for c in country.split(",") if c.strip()] if country else ["UK", "CL", "AR", "MX", "ES"]
    countries = cc_list_for_breakdown
    for cc in countries:
        q = db.query(func.count(Article.id)).filter(
            Article.review_status == "pending",
            Article.title.isnot(None),
            Article.country == cc,
        )
        if batch_size and cc == (country or cc):
            n = min(q.scalar() or 0, batch_size)
        else:
            n = q.scalar() or 0
        breakdown[cc] = {
            "articles": n,
            "cost_usd": round(n * COST_PER_ARTICLE, 2),
        }

    _llm_log("━" * 50)
    _llm_log("  COST ESTIMATE — LLM Classification")
    _llm_log("━" * 50)
    for cc, d in breakdown.items():
        _llm_log(f"  {cc}: {d['articles']:,} articulos → ${d['cost_usd']:.2f}")
    _llm_log("━" * 50)
    _llm_log(f"  TOTAL: {total:,} articulos → ${total_cost:.2f} USD")
    _llm_log(f"  Modelo: Claude claude-opus-4-5 (~$0.0203/articulo)")
    _llm_log("━" * 50)

    return {
        "total_articles": total,
        "total_cost_usd": round(total_cost, 2),
        "cost_per_article": round(COST_PER_ARTICLE, 4),
        "breakdown": breakdown,
    }
