"""
jobs.py — scheduled background jobs (APScheduler):

  daily_fetch_job      08:00 UTC  fetch yesterday's articles from all configured sources,
                                  LLM-classify pending articles, enrich confirmed ones
  intraday classify    13:00, 19:00, 01:00 UTC
  auto_approve_job     every 30 min — confirm/reject LLM decisions that are confident enough
  weekly_audit_job     Sunday 09:00 UTC — re-classify a random sample and measure drift
  embeddings_backfill  02:00 UTC — embed confirmed articles, then link orphan articles to cases
"""
from __future__ import annotations
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from typing import Any, Dict, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from .database import SessionLocal, save_article
from .guardian import fetch_day
from .log_buffer import log as _log


def _jobs_log(msg, level="info"):
    _log(msg, source="jobs", level=level)


scheduler = AsyncIOScheduler(timezone="UTC")

AUTO_APPROVE_MIN_CONFIDENCE = 0.88
FETCH_WINDOW_DAYS = 6  # RSS feeds are relevance-ranked; dedup in save_article drops repeats


# ── Daily fetch ───────────────────────────────────────────────────────────────

def _run_fetch(day: date) -> Dict[str, Any]:
    from .countries import ACTIVE_COUNTRIES, fetch_gdelt_country, fetch_rss_country

    db = SessionLocal()
    counts = {"saved": 0, "total": 0}

    def _process(article: dict) -> None:
        counts["total"] += 1
        text = f"{article.get('title', '')} {article.get('content_snippet', '')}".strip()
        if not text:
            return
        article["review_status"] = "pending"
        article["ml_score"] = None
        if save_article(db, article):
            counts["saved"] += 1

    def _fetch_country(cc: str):
        articles = list(fetch_rss_country(cc))
        articles += list(fetch_gdelt_country(cc, day - timedelta(days=FETCH_WINDOW_DAYS), day))
        return cc, articles

    try:
        _jobs_log(f"[jobs] === Daily scan for {day} ===")
        _jobs_log(f"[jobs] Fetching Guardian for {day}…")
        try:
            for article in fetch_day(day):
                _process(article)
        except Exception as exc:
            _jobs_log(f"[jobs] Guardian fetch failed: {exc}", level="error")

        countries = list(ACTIVE_COUNTRIES)
        if countries:
            with ThreadPoolExecutor(max_workers=len(countries)) as pool:
                futures = {pool.submit(_fetch_country, cc): cc for cc in countries}
                for future in as_completed(futures):
                    try:
                        cc, articles = future.result()
                    except Exception as exc:
                        _jobs_log(f"[jobs] {futures[future]}: fetch failed — {exc}", level="error")
                        continue
                    window_start = day - timedelta(days=FETCH_WINDOW_DAYS)
                    n = 0
                    for article in articles:
                        pub = article.get("published_date")
                        if pub and pub >= window_start:
                            _process(article)
                            n += 1
                    _jobs_log(f"[jobs] {cc}: {n} articles processed")
    finally:
        db.close()

    _jobs_log(f"[jobs] {day} → fetched={counts['total']} saved={counts['saved']}", level="success")
    return {"date": str(day), "fetched": counts["total"], "saved": counts["saved"]}


def _run_llm_classification(country: Optional[str] = None, batch_size: int = 100) -> Dict[str, Any]:
    """Run the LLM cascade over pending articles."""
    from .llm_classifier import classify_pending_articles
    db = SessionLocal()
    try:
        _jobs_log(f"[jobs] Starting LLM classification (country={country or 'all'}, batch={batch_size})…")
        summary = classify_pending_articles(db, country=country, batch_size=batch_size, auto_confirm=True)
        _jobs_log(f"[jobs] LLM classification done: {summary}", level="success")
        return summary
    except Exception as e:
        _jobs_log(f"[jobs] LLM classification error: {e}", level="error")
        return {"error": str(e)}
    finally:
        db.close()


async def daily_fetch_job():
    """Fetch, classify, enrich."""
    day = date.today() - timedelta(days=1)
    loop = asyncio.get_event_loop()

    fetch_result = await loop.run_in_executor(None, _run_fetch, day)
    _jobs_log(f"[jobs] Fetch complete: {fetch_result}", level="success")

    llm_result = await loop.run_in_executor(None, _run_llm_classification)
    _jobs_log(f"[jobs] LLM classify complete: {llm_result}", level="success")

    try:
        from .enrichment import enrich_pending_articles
        db = SessionLocal()
        try:
            _jobs_log(f"[jobs] Enrichment complete: {enrich_pending_articles(db, limit=30, delay=1.0)}", level="success")
        finally:
            db.close()
    except Exception as e:
        _jobs_log(f"[jobs] Enrichment error: {e}", level="error")


def manual_fetch(day: date) -> Dict[str, Any]:
    """Fetch a specific day, then classify what arrived."""
    fetch_result = _run_fetch(day)
    return {**fetch_result, "llm": _run_llm_classification}


def manual_llm_classify(country: Optional[str] = None, batch_size: int = 100) -> Dict[str, Any]:
    return _run_llm_classification(country, batch_size)


# ── Periodic jobs ─────────────────────────────────────────────────────────────

async def auto_approve_job():
    """Confirm/reject articles the LLM classified >2h ago with high confidence."""
    from .database import Article, POSITIVE_TYPE_IDS
    db = SessionLocal()
    try:
        cutoff = datetime.utcnow() - timedelta(hours=2)
        pending = db.query(Article).filter(
            Article.review_status == "pending",
            Article.event_type.isnot(None),
            Article.llm_classified_at.isnot(None),
            Article.llm_classified_at < cutoff,
            Article.ml_score >= AUTO_APPROVE_MIN_CONFIDENCE,
        ).all()
        if not pending:
            return
        confirmed = rejected = 0
        for art in pending:
            positive = art.event_type in POSITIVE_TYPE_IDS
            art.review_status = "confirmed" if positive else "rejected"
            art.human_label = 1 if positive else 0
            art.reviewed_by = "auto-approve"
            art.reviewed_at = datetime.utcnow()
            if positive:
                confirmed += 1
                try:
                    from .embeddings import embed_article
                    embed_article(db, art)
                except Exception as e:
                    _jobs_log(f"[auto-approve] embed_article failed (non-fatal): {e}", level="error")
            else:
                rejected += 1
        db.commit()
        _jobs_log(f"[auto-approve] confirmed={confirmed} rejected={rejected} "
                  f"(min_confidence={AUTO_APPROVE_MIN_CONFIDENCE})", level="success")
    except Exception as e:
        db.rollback()
        _jobs_log(f"[auto-approve] Error: {e}", level="error")
    finally:
        db.close()


async def _intraday_classify_job():
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _run_llm_classification)
    _jobs_log(f"[intraday-classify] {result}", level="success" if "error" not in result else "error")


async def weekly_audit_job():
    """Re-classify a random sample and report the discrepancy rate (drift)."""
    from .quality_audit import audit_sample, flag_discrepant_articles
    db = SessionLocal()
    try:
        report = audit_sample(db, sample_size=50)
        if report.get("high_severity", 0) > 0:
            flagged = flag_discrepant_articles(db, report["cases"])
            _jobs_log(f"[audit] {flagged} high-severity articles flagged for human review.", level="error")
        _jobs_log(
            f"[audit] checked={report['checked']}, discrepancies={report['discrepancies']} "
            f"({report['drift_pct']}%), cost=${report['cost_usd']:.4f}",
            level="error" if report.get("alert") else "success",
        )
    except Exception as e:
        _jobs_log(f"[audit] Error: {e}", level="error")
    finally:
        db.close()


async def embeddings_backfill_job():
    """Embed confirmed articles, then link orphan (case-less) articles to cases."""
    from .embeddings import is_backfill_running, run_embeddings_backfill
    if is_backfill_running():
        _jobs_log("[embeddings-backfill] Skipped — a run is already in progress.")
        return
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, run_embeddings_backfill, 900, 0.5, None)

    from .cases import link_all_unlinked
    from .database import Article as _Article, Case as _Case
    db = SessionLocal()
    try:
        result = await loop.run_in_executor(None, link_all_unlinked, db, _Case, _Article, None)
        _jobs_log(f"[embeddings-backfill] link_all_unlinked: {result}")
    except Exception as e:
        _jobs_log(f"[embeddings-backfill] link_all_unlinked failed (non-fatal): {e}", level="error")
    finally:
        db.close()


# ── Scheduler ─────────────────────────────────────────────────────────────────

def _add(func, trigger, job_id, grace, **kw):
    scheduler.add_job(func, trigger, id=job_id, replace_existing=True,
                      misfire_grace_time=grace, coalesce=True, **kw)


def start_scheduler():
    _add(daily_fetch_job, CronTrigger(hour=8, minute=0), "daily_fetch", 3600, max_instances=1)
    _add(auto_approve_job, CronTrigger(minute="*/30"), "auto_approve", 1800)
    for hour in (13, 19, 1):
        _add(_intraday_classify_job, CronTrigger(hour=hour, minute=0), f"intraday_classify_{hour:02d}", 3600)
    _add(weekly_audit_job, CronTrigger(day_of_week="sun", hour=9, minute=0), "weekly_audit", 3600)
    _add(embeddings_backfill_job, CronTrigger(hour=2, minute=0), "embeddings_backfill", 3600, max_instances=1)
    scheduler.start()
    _jobs_log("[scheduler] started: daily fetch 08:00 UTC, auto-approve 30 min, intraday classify, "
              "weekly audit Sun 09:00 UTC, embeddings backfill 02:00 UTC")


def get_scheduler_status():
    if not scheduler.running:
        return {"status": "NOT RUNNING", "jobs": []}
    jobs = [{
        "id": job.id,
        "next_run": str(job.next_run_time) if job.next_run_time else "Never",
        "trigger": str(job.trigger),
    } for job in scheduler.get_jobs()]
    return {"status": "RUNNING", "job_count": len(jobs), "jobs": jobs}
