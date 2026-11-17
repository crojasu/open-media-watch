"""
fmedia/embeddings.py — RAG embedding service
----------------------------------------------------------------------------------------
embed_text(text) -> list[float] | None
    Primary:  Gemini `text-embedding-004` (768 dims, free tier) — same lazy-client
              pattern as fmedia/gemini_triage.py.
    Fallback: OpenAI `text-embedding-3-small`, truncated to 768 dims via the
              `dimensions` param — same cascade *style* as llm_classifier.classify_article()
              (try primary, log + fall through to secondary on any error).
    No API key configured for either provider → returns None and logs (tests mock this).

build_embedding_text(title, event_summary, content_snippet, full_text) -> str
    `title + " " + (event_summary or full_text or content_snippet or "")`,
    truncated to MAX_TEXT_CHARS (2000).

VectorStore — two backends selected by DB dialect:
    PgVectorStore — Postgres + pgvector. Cosine via the `<=>` operator, raw
                      parameterized SQL (the `embedding vector(768)` column is
                      NOT ORM-mapped — see fmedia/database.py Article docstring
 — so all reads/writes go through text() SQL here).
    BruteForceStore — SQLite (dev). Cosine similarity computed in pure Python
                      over the `embedding_json` TEXT column (JSON-encoded
                      list[float]) — viable because the dev corpus is small.

get_vector_store(db) -> VectorStore  factory, picks backend from the bound
    engine's dialect name.

embed_article(db, article) -> bool
    Convenience wrapper used by the confirm-article hooks (jobs.py auto-approve,
    main.py human review) and by scripts/backfill_embeddings.py: builds the
    embedding text, calls embed_text(), and persists it via the right VectorStore
    backend. Catches everything and returns False on any failure — NEVER raises,
    so it can never break the confirm-article flow that calls it.
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Dict, List, Optional

from sqlalchemy import extract, text
from sqlalchemy.orm import Session

from .log_buffer import log as _log


def _emb_log(msg: str, level: str = "info") -> None:
    _log(msg, source="embeddings", level=level)


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

GEMINI_EMBED_MODEL = "gemini-embedding-001"  # text-embedding-004 retirado del API (ene-2026); 768 dims vía output_dimensionality
OPENAI_EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 768
MAX_TEXT_CHARS = 2000


# ── Text preparation ───────────────────────────────────────────────────────────

def build_embedding_text(
    title: Optional[str],
    event_summary: Optional[str] = None,
    content_snippet: Optional[str] = None,
    full_text: Optional[str] = None,
) -> str:
    """title + ' ' + body, truncated to 2000 chars.

    Body priority: event_summary (LLM-curated, most relevant signal when
    present) > full_text (the scraped article body, has the
    circumstantial detail — location, manner of death, relationship — that
    matters most for matching an UNNAMED victim across articles published
    months apart, since there's no name to match on) > content_snippet
    (just the RSS title+summary, ~80-100 chars, weakest signal)."""
    body = event_summary or full_text or content_snippet or ""
    combined = f"{title or ''} {body}".strip()
    return combined[:MAX_TEXT_CHARS]


# ── Provider cascade (Gemini primary, OpenAI fallback) ────────────────────────

def _embed_gemini(input_text: str) -> Optional[List[float]]:
    if not GEMINI_API_KEY:
        return None
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.embed_content(
            model=GEMINI_EMBED_MODEL,
            contents=input_text,
            config=types.EmbedContentConfig(output_dimensionality=EMBED_DIM),
        )
        return list(response.embeddings[0].values)
    except Exception as e:
        _emb_log(f"[embeddings] Gemini embed error: {e}", level="error")
        return None


def _embed_openai(input_text: str) -> Optional[List[float]]:
    if not OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.embeddings.create(
            model=OPENAI_EMBED_MODEL,
            input=input_text,
            dimensions=EMBED_DIM,
        )
        return list(response.data[0].embedding)
    except Exception as e:
        _emb_log(f"[embeddings] OpenAI embed error: {e}", level="error")
        return None


def embed_text(input_text: str) -> Optional[List[float]]:
    """Gemini text-embedding-004 (primary) → OpenAI text-embedding-3-small (fallback).
    Returns None (and logs) if both fail or neither API key is configured."""
    if not input_text or not input_text.strip():
        return None
    result = _embed_gemini(input_text)
    if result is None:
        _emb_log("[embeddings] Gemini unavailable — trying OpenAI…", level="error")
        result = _embed_openai(input_text)
    if result is None:
        _emb_log("[embeddings] All embedding providers failed or unconfigured.", level="error")
    return result


# ── Cosine similarity (pure python, no numpy dependency needed for this) ──────

def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _vector_literal(vec: List[float]) -> str:
    """Postgres array-literal text for a pgvector column, e.g. '[0.1,0.2,...]'."""
    return "[" + ",".join(repr(float(v)) for v in vec) + "]"


def _parse_vector_literal(s: Optional[str]) -> Optional[List[float]]:
    """Inverse of _vector_literal() — parses a pgvector `::text` cast like
    '[0.1,0.2,0.3]' back into a list[float]. Used by PgVectorStore.vectors_for_status(). Returns
    None on any malformed input; a bad row should never crash a discovery scan."""
    if not s:
        return None
    try:
        inner = s.strip().strip("[]")
        if not inner:
            return []
        return [float(x) for x in inner.split(",")]
    except (ValueError, AttributeError):
        return None


# ── VectorStore backends ──────────────────────────────────────────────────

class VectorStore:
    def search(
        self,
        query_vector: List[float],
        k: int = 10,
        country: Optional[str] = None,
        year: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Returns [{"article": Article, "score": float}, ...] ordered best-first."""
        raise NotImplementedError

    def save_embedding(self, article, vector: List[float]) -> None:
        raise NotImplementedError

    def vectors_for_status(
        self,
        review_status: str,
        country: Optional[str] = None,
        limit: Optional[int] = None,
        local_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """ — return
        [{"article": Article, "vector": list[float]}, ...] for every article with
        the given review_status that already has an embedding, ordered by id.

        Used by fmedia/semantic_discovery.py for:
          - the CONFIRMED-article centroid (review_status="confirmed",
            local_only=True — excludes foreign incidents, same as the nearest-
            neighbor search() above)
          - scoring PENDING candidates (review_status="pending")

        This is a raw vector dump, not a similarity ranking — see search() for
        nearest-neighbor lookup against a query vector."""
        raise NotImplementedError


class BruteForceStore(VectorStore):
    """SQLite backend: cosine similarity over embedding_json, computed in Python.
    Viable because the dev/test corpus is small."""

    def __init__(self, db: Session):
        self.db = db

    def save_embedding(self, article, vector: List[float]) -> None:
        article.embedding_json = json.dumps(vector)
        self.db.add(article)

    def search(
        self,
        query_vector: List[float],
        k: int = 10,
        country: Optional[str] = None,
        year: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        from .database import Article, local_incident_filter

        q = self.db.query(Article).filter(
            Article.review_status == "confirmed",
            Article.embedding_json.isnot(None),
            local_incident_filter(),
        )
        if country:
            q = q.filter(Article.country == country)
        if year:
            q = q.filter(extract("year", Article.published_date) == year)

        scored = []
        for art in q.all():
            try:
                vec = json.loads(art.embedding_json)
            except (TypeError, ValueError):
                continue
            score = cosine_similarity(query_vector, vec)
            scored.append((score, art))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [{"article": art, "score": score} for score, art in scored[:k]]

    def vectors_for_status(
        self,
        review_status: str,
        country: Optional[str] = None,
        limit: Optional[int] = None,
        local_only: bool = False,
    ) -> List[Dict[str, Any]]:
        from .database import Article, local_incident_filter

        q = self.db.query(Article).filter(
            Article.review_status == review_status,
            Article.embedding_json.isnot(None),
        )
        if local_only:
            q = q.filter(local_incident_filter())
        if country:
            q = q.filter(Article.country == country)
        q = q.order_by(Article.id.asc())
        if limit:
            q = q.limit(limit)

        out = []
        for art in q.all():
            try:
                vec = json.loads(art.embedding_json)
            except (TypeError, ValueError):
                continue
            out.append({"article": art, "vector": vec})
        return out


class PgVectorStore(VectorStore):
    """Postgres + pgvector backend. The `embedding vector(768)` column is NOT
    ORM-mapped (see fmedia/database.py Article docstring) — everything here is
    raw parameterized SQL, cosine distance via the `<=>` operator."""

    def __init__(self, db: Session):
        self.db = db

    def save_embedding(self, article, vector: List[float]) -> None:
        self.db.execute(
            text("UPDATE articles SET embedding = CAST(:vec AS vector) WHERE id = :id"),
            {"vec": _vector_literal(vector), "id": article.id},
        )

    def search(
        self,
        query_vector: List[float],
        k: int = 10,
        country: Optional[str] = None,
        year: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        from .database import Article, LOCAL_INCIDENT_SQL

        sql = text(
            f"""
            SELECT id, (embedding <=> CAST(:qvec AS vector)) AS distance
            FROM articles
            WHERE review_status = 'confirmed'
              AND embedding IS NOT NULL
              AND {LOCAL_INCIDENT_SQL}
              AND (:country IS NULL OR country = :country)
              AND (:year IS NULL OR EXTRACT(YEAR FROM published_date) = :year)
            ORDER BY embedding <=> CAST(:qvec AS vector)
            LIMIT :k
            """
        )
        rows = self.db.execute(
            sql,
            {"qvec": _vector_literal(query_vector), "country": country, "year": year, "k": k},
        ).fetchall()
        if not rows:
            return []
        ids = [r.id for r in rows]
        distance_by_id = {r.id: r.distance for r in rows}
        arts = self.db.query(Article).filter(Article.id.in_(ids)).all()
        by_id = {a.id: a for a in arts}
        # pgvector `<=>` is cosine DISTANCE (0 = identical) → similarity = 1 - distance
        results = [
            {"article": by_id[i], "score": 1.0 - distance_by_id[i]}
            for i in ids
            if i in by_id
        ]
        results.sort(key=lambda r: r["score"], reverse=True)
        return results

    def vectors_for_status(
        self,
        review_status: str,
        country: Optional[str] = None,
        limit: Optional[int] = None,
        local_only: bool = False,
    ) -> List[Dict[str, Any]]:
        from .database import Article, LOCAL_INCIDENT_SQL

        # search() above never needs the raw vector back in Python — Postgres
        # does the distance math and only the score crosses the wire. Centroid
        # computation genuinely needs every raw vector, so this casts
        # the pgvector column to text (pgvector supports `::text`, formatting as
        # '[0.1,0.2,...]' — the same literal shape _vector_literal() writes) and
        # parses it back with _parse_vector_literal() below. `local_only` is a
        # static Python bool (never user input), so it's safe to splice into the
        # SQL structurally rather than bind it as a parameter.
        local_clause = f"AND {LOCAL_INCIDENT_SQL}" if local_only else ""
        limit_clause = "LIMIT :limit" if limit else ""
        sql = text(
            f"""
            SELECT id, embedding::text AS emb_text
            FROM articles
            WHERE review_status = :status
              AND embedding IS NOT NULL
              {local_clause}
              AND (:country IS NULL OR country = :country)
            ORDER BY id ASC
            {limit_clause}
            """
        )
        params: Dict[str, Any] = {"status": review_status, "country": country}
        if limit:
            params["limit"] = limit
        rows = self.db.execute(sql, params).fetchall()
        if not rows:
            return []

        ids = [r.id for r in rows]
        vec_by_id = {r.id: _parse_vector_literal(r.emb_text) for r in rows}
        arts = self.db.query(Article).filter(Article.id.in_(ids)).all()
        by_id = {a.id: a for a in arts}
        return [
            {"article": by_id[i], "vector": vec_by_id[i]}
            for i in ids
            if i in by_id and vec_by_id[i] is not None
        ]


def get_vector_store(db: Session) -> VectorStore:
    is_postgres = db.get_bind().dialect.name == "postgresql"
    return PgVectorStore(db) if is_postgres else BruteForceStore(db)


# ── Confirm-article hook entry point ────────────────────────────────

def embed_article(db: Session, article) -> bool:
    """Compute + persist the embedding for one (already-confirmed) article.

    Used by:
      - fmedia/jobs.py  auto_approve_job()  (scheduler auto-approve path)
      - fmedia/main.py  do_review()         (human review confirm path)
      - scripts/backfill_embeddings.py      (batch backfill)

    NEVER raises — returns True/False and logs. This is intentional: embedding
    failures must never break the article confirmation flow that calls this.
    """
    try:
        input_text = build_embedding_text(
            getattr(article, "title", None),
            getattr(article, "event_summary", None),
            getattr(article, "content_snippet", None),
            getattr(article, "full_text", None),
        )
        vector = embed_text(input_text)
        if vector is None:
            return False
        store = get_vector_store(db)
        store.save_embedding(article, vector)
        db.commit()
        return True
    except Exception as e:
        _emb_log(
            f"[embeddings] embed_article failed for article "
            f"{getattr(article, 'id', '?')}: {e}",
            level="error",
        )
        try:
            db.rollback()
        except Exception:
            pass
        return False


# ── Admin-triggered manual backfill job ───────────────────
# Deliberately NOT on the scheduler (unlike the daily fetch / auto-approve
# jobs in jobs.py) — Gemini's free tier is 1,000 requests/day SHARED with
# gemini_triage.py's classification triage, so an unattended daily job here
# would compete with triage for the same cap. An operator triggers this from
# /admin/backfill when they want to spend that day's embedding budget.

_backfill_state: Dict[str, Any] = {"running": False, "last_result": None}


def is_backfill_running() -> bool:
    return _backfill_state["running"]


def last_backfill_result() -> Optional[Dict[str, Any]]:
    return _backfill_state["last_result"]


def get_embedding_counts(db: Session) -> Dict[str, int]:
    """Live counts for the admin panel — this IS the backfill's progress,
    no separate progress file needed (unlike fmedia/backfill.py's article
    backfill): "remaining" is just a fresh COUNT of confirmed articles with
    no embedding yet."""
    from .database import Article
    is_pg = db.get_bind().dialect.name == "postgresql"
    confirmed = db.query(Article).filter(Article.review_status == "confirmed")
    total = confirmed.count()
    missing = (
        confirmed.filter(text("embedding IS NULL")).count() if is_pg
        else confirmed.filter(Article.embedding_json.is_(None)).count()
    )
    return {"confirmed_total": total, "missing": missing, "embedded": total - missing}


def run_embeddings_backfill(
    limit: Optional[int] = None, sleep_s: float = 0.5, country: Optional[str] = None,
) -> Dict[str, int]:
    """Embeds confirmed articles with no embedding yet, oldest id first —
    the same resumable "WHERE embedding IS NULL" query scripts/
    backfill_embeddings.py uses, so re-running (here or via the CLI script)
    always picks up where the last run left off, never repeats or reorders
    what's already done.

    `limit` should stay under Gemini's free-tier daily cap (1,000/day, see
    module docstring above) — the admin panel defaults to a conservative
    number, leaving headroom for triage's share of the same cap.
    """
    from .database import Article, SessionLocal

    _backfill_state["running"] = True
    db = SessionLocal()
    stats = {"total": 0, "embedded": 0, "failed": 0}
    try:
        is_pg = db.get_bind().dialect.name == "postgresql"
        q = db.query(Article).filter(Article.review_status == "confirmed")
        q = q.filter(text("embedding IS NULL")) if is_pg else q.filter(Article.embedding_json.is_(None))
        if country:
            q = q.filter(Article.country == country.upper())
        q = q.order_by(Article.id.asc())
        if limit:
            q = q.limit(limit)
        articles = q.all()
        stats["total"] = len(articles)
        _emb_log(f"[backfill] {len(articles)} article(s) to process…")

        for i, art in enumerate(articles):
            ok = embed_article(db, art)
            stats["embedded" if ok else "failed"] += 1
            if (i + 1) % 25 == 0 or (i + 1) == len(articles):
                _emb_log(f"[backfill] [{i + 1}/{len(articles)}] embedded={stats['embedded']} failed={stats['failed']}")
            if i + 1 < len(articles):
                time.sleep(sleep_s)

        _emb_log(f"[backfill] Done. embedded={stats['embedded']} failed={stats['failed']} total={stats['total']}")
        return stats
    finally:
        db.close()
        _backfill_state["running"] = False
        _backfill_state["last_result"] = stats
