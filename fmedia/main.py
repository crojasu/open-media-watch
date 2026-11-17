"""main.py — Open Media Watch: public read-only REST API (FastAPI).

Starts the scheduler (see jobs.py) and serves:

  GET  /health
  GET  /api/articles            confirmed articles (public fields only)
  GET  /api/timeline            monthly counts
  GET  /api/types               counts by category
  GET  /api/relaciones          counts by secondary label 1
  GET  /api/motivos             counts by secondary label 2
  GET  /api/types-by-month
  GET  /api/sankey              category x label cross-tabulation
  GET  /api/dossiers            published dossiers (see dossiers.py for guardrails)
  GET  /api/dossiers/by-slug/{slug}
  POST/PATCH /api/dossiers/...  dossier write API (HTTP Basic, reviewer credentials)

This repository ships no web UI and no admin panel.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func
from sqlalchemy.orm import Session

from .database import (
    Article, get_articles_paginated, get_cross_tabulation, get_date_range,
    get_db, get_monthly_counts, get_motivo_counts, get_relacion_counts,
    get_total_count, get_type_counts, get_type_counts_by_month, init_db,
)
from .dossiers import router as dossiers_router
from .jobs import scheduler, start_scheduler
from .log_buffer import log as _log
from .trusted_proxy import TrustedClientIPMiddleware

load_dotenv()

_REQUIRED_ENV_VARS = ["REVIEW_USERS", "ANTHROPIC_API_KEY", "DATABASE_URL"]
PER_PAGE = 20



def _validate_env() -> None:
    missing = [v for v in _REQUIRED_ENV_VARS if not os.getenv(v)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _validate_env()
    init_db()
    _log("[startup] Database ready.", source="app")
    start_scheduler()
    yield
    scheduler.shutdown(wait=False)


app = FastAPI(title="Open Media Watch", version="0.1.0", lifespan=lifespan,
              docs_url="/docs", redoc_url=None)

app.add_middleware(TrustedClientIPMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in os.getenv("CORS_ORIGINS", "").split(",") if o],
    allow_methods=["GET", "POST", "PATCH"],
    allow_headers=["*"],
)
app.include_router(dossiers_router)


def _fmt_public(a: Article) -> dict:
    """Article as exposed by the public API: no reviewer identity, status or labels."""
    return {
        "id": a.id,
        "title": a.title,
        "web_url": a.web_url,
        "section": a.section,
        "published_date": str(a.published_date),
        "source": a.source,
        "country": a.country,
        "snippet": a.content_snippet,
        "event_type": a.event_type,
        "event_relation": a.event_relation,
        "event_motive": a.event_motive,
    }


@app.get("/health")
def health(db: Session = Depends(get_db)):
    counts = {
        st: db.query(func.count(Article.id)).filter(Article.review_status == st).scalar()
        for st in ("pending", "confirmed", "rejected")
    }
    counts["total"] = db.query(func.count(Article.id)).scalar()
    return {"status": "ok", "article_counts": counts}


@app.get("/api/articles")
def api_articles(
    db: Session = Depends(get_db),
    page: int = Query(1, ge=1),
    year: Optional[int] = Query(None, ge=2000, le=2100),
    month: Optional[int] = Query(None, ge=1, le=12),
    country: Optional[str] = Query(None, min_length=2, max_length=2),
    type: Optional[str] = Query(None),
    relacion: Optional[str] = Query(None),
    motivo: Optional[str] = Query(None),
):
    result = get_articles_paginated(
        db, page=page, per_page=PER_PAGE, year=year, month=month,
        country=country.upper() if country else None,
        event_type=type, event_relation=relacion, event_motive=motivo,
    )
    return {
        "total": result["total"], "page": result["page"], "per_page": result["per_page"],
        "items": [_fmt_public(a) for a in result["items"]],
    }


@app.get("/api/timeline")
def api_timeline(db: Session = Depends(get_db), country: Optional[str] = Query(None)):
    cc = country.upper() if country else None
    return {
        "monthly_counts": get_monthly_counts(db, country=cc),
        "total": get_total_count(db, country=cc),
        "date_range": get_date_range(db, country=cc),
    }


@app.get("/api/types")
def api_types(db: Session = Depends(get_db), country: Optional[str] = Query(None)):
    return get_type_counts(db, country=country.upper() if country else None)


@app.get("/api/relaciones")
def api_relaciones(db: Session = Depends(get_db), country: Optional[str] = Query(None)):
    return get_relacion_counts(db, country=country.upper() if country else None)


@app.get("/api/motivos")
def api_motivos(db: Session = Depends(get_db), country: Optional[str] = Query(None)):
    return get_motivo_counts(db, country=country.upper() if country else None)


@app.get("/api/types-by-month")
def api_types_by_month(db: Session = Depends(get_db), country: Optional[str] = Query(None)):
    return get_type_counts_by_month(db, country=country.upper() if country else None)


@app.get("/api/sankey")
def api_sankey(db: Session = Depends(get_db), country: Optional[str] = Query(None)):
    return get_cross_tabulation(db, country=country.upper() if country else None)
