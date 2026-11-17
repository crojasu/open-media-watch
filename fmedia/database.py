"""
database.py
-----------
SQLite persistence. Articles flow through review_status:
  pending   → classified by ML, waiting for human review
  confirmed → human said YES → appears in public timeline
  rejected  → human said NO  → stored for training, never shown publicly
"""
from __future__ import annotations
import os
import re as _re_mod
import unicodedata as _unicodedata
from datetime import date, datetime
from typing import List, Optional
from sqlalchemy import (
    Column, Date, DateTime, Integer, String, Text, Boolean, Float,
    ForeignKey, Index, UniqueConstraint, CheckConstraint,
    create_engine, desc, func, extract, case, cast, text, or_, inspect
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker, relationship, validates
from sqlalchemy.pool import StaticPool

_db_url = os.getenv("DATABASE_URL", "sqlite:///data/omw.db")
# Some hosts provide postgres:// but SQLAlchemy needs postgresql://
DB_PATH = _db_url.replace("postgres://", "postgresql://", 1)

# SQLite needs StaticPool to avoid "QueuePool limit reached" errors.
# For Postgres (production), pool_pre_ping keeps connections alive.
if DB_PATH.startswith("sqlite"):
    engine = create_engine(
        DB_PATH,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
else:
    engine = create_engine(DB_PATH, pool_pre_ping=True)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Base(DeclarativeBase):
    pass


class Article(Base):
    __tablename__ = "articles"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    guardian_id      = Column(String(500), unique=True, nullable=False)
    title            = Column(Text, nullable=False)
    web_url          = Column(Text)
    thumbnail        = Column(Text)
    section          = Column(String(200))
    published_date   = Column(Date, nullable=False)
    fetched_at       = Column(DateTime, default=datetime.utcnow)
    country          = Column(String(10), default="UK")
    source           = Column(String(100), default="The Guardian")
    content_snippet  = Column(Text)
    # Full scraped article body — enrichment.py's scrape_article()
    # used to fetch this and throw it away after one re-extraction call.
    # content_snippet is really just the RSS title+summary (~80-100 chars),
    # which starved victim_name/incident_date/incident_location extraction
    # and case-linking. Nullable — only populated going forward, not
    # backfilled for already-confirmed articles.
    full_text        = Column(Text, nullable=True)
    # Enrichment attempt tracking — enrichment.py used to retry
    # the same failing articles every day. NULL attempts == never tried.
    enrichment_attempts        = Column(Integer, nullable=True, default=0)
    enrichment_last_attempt_at = Column(DateTime, nullable=True)
    enrichment_last_result     = Column(String(40), nullable=True)
    # Review
    review_status    = Column(String(20), default="pending")  # pending/confirmed/rejected
    reviewed_by      = Column(String(100), nullable=True)
    reviewed_at      = Column(DateTime, nullable=True)
    human_label      = Column(Integer, nullable=True)  # 1=relevant incident 0=not, used for retraining
    # Typology (LLM classifier output)
    event_type     = Column(String(50), nullable=True)
    event_relation = Column(String(50), nullable=True)
    event_motive   = Column(String(50), nullable=True)
    ml_score          = Column(Float, nullable=True)
    llm_classified_at = Column(DateTime, nullable=True)

    # ── Victim / case extraction (LLM) ──
    victim_name              = Column(String, nullable=True)
    victim_age               = Column(Integer, nullable=True)
    victim_nationality       = Column(String, nullable=True)
    incident_location        = Column(String, nullable=True)
    incident_region          = Column(String, nullable=True)
    incident_date_extracted  = Column(Date, nullable=True)
    perpetrator_name         = Column(String, nullable=True)
    perpetrator_age          = Column(Integer, nullable=True)
    perpetrator_status       = Column(String, nullable=True)
    had_prior_complaints     = Column(Boolean, nullable=True)
    children_orphaned        = Column(Integer, nullable=True)
    weapon                   = Column(String, nullable=True)
    event_summary         = Column(Text, nullable=True)

    # ── LLM cost tracking ──
    llm_model_used    = Column(String(100), nullable=True)
    llm_tokens_in     = Column(Integer, nullable=True)
    llm_tokens_out    = Column(Integer, nullable=True)
    llm_cost_usd      = Column(Float, nullable=True)

    # ── Origin tracking ──
    origin = Column(String(20), default="daily")

    # ── Cross-source deduplication ──
    content_fingerprint = Column(String(64), nullable=True, index=True)

    # ── Case grouping ──
    case_id = Column(Integer, ForeignKey("cases.id"), nullable=True)

    # ── RAG embeddings ──
    # embedding_json is a REAL ORM column that exists identically on SQLite and
    # Postgres (create_all() creates it as TEXT everywhere — this is what keeps
    # `tests/conftest.py`'s SQLite create_all() unaffected by this feature).
    # It stores a JSON-encoded list[float] and backs the SQLite BruteForceStore.
    #
    # On Postgres there is ALSO a separate `embedding vector(768)` column, added
    # by alembic/versions (migration 001) and managed entirely via raw SQL in
    # fmedia/embeddings.py (PgVectorStore) — it is deliberately NOT mapped here.
    # Reasoning: SQLAlchemy has no single column type that compiles cleanly to
    # both SQLite and Postgres DDL for a 768-dim vector; rather than pull in a
    # TypeDecorator with dialect-specific branches, the vector column is fully
    # out-of-ORM (raw SQL start to finish), and this Text column is the only one
    # `Base.metadata.create_all()` ever touches. Simplest thing that works in both.
    embedding_json = Column(Text, nullable=True)

    # ── Incident country (, — wrong-country
    # incident fix, 31-jul) ──────────────────────────────────────────────────
    # ISO-2 of the country where the relevant incident ACTUALLY happened, as determined
    # by the LLM classifier from the article text. This is deliberately
    # DIFFERENT from `country` (the monitored media outlet's country): a
    # Guardian UK article can report a relevant incident that happened in India, and
    # `country` stays "UK" (that's the source being monitored) while
    # `incident_country` would be "IN". NULL means "not determined by the LLM"
    # OR "same as `country`" — both read as LOCAL by local_incident_filter()
    # below. Only a genuinely different ISO-2 code marks an article as
    # foreign — still `confirmed` (it's a real relevant incident), but excluded from
    # every country-scoped counting/listing surface and never linked to a
    # local Case (see fmedia/cases.py's foreign-incident guard).
    incident_country = Column(String(2), nullable=True)

    __table_args__ = (
        Index("ix_articles_country_status_date", "country", "review_status", "published_date"),
        Index("ix_articles_status_classified_at", "review_status", "llm_classified_at"),
        Index("ix_articles_country_type_date", "country", "event_type", "published_date"),
        Index("ix_articles_published_date", "published_date"),
    )


# ── Valid typology values ─────────────────────────────────────────────────────
from .taxonomy import POSITIVE_TYPES as POSITIVE_TYPE_IDS  # noqa: E402  (loaded from the taxonomy file)


# ── Local-incident filter ─────────────────
# Single reusable predicate shared by every country-scoped counting/listing
# surface (stats endpoints, article listings, and implicitly the case-linker,
# which short-circuits before ever running a query — see cases.py). Deliberately
# NOT parameterized by a query `country=...` argument: it compares each row's
# own `incident_country` against its OWN `country` column, so it composes with
# any existing country filter instead of replacing it.
#
# `Article.country` holds the app's INTERNAL 2-letter code, which is not always
# the real ISO 3166-1 alpha-2 code the LLM writes into `incident_country`: the
# United Kingdom is tracked here as "UK" but classified as "GB". So the two
# columns are never compared raw — `country` is normalized through this map
# first. Add an entry if another non-ISO internal code is ever introduced.
COUNTRY_CODE_TO_ISO = {"UK": "GB"}


def to_iso_country(code: Optional[str]) -> Optional[str]:
    """Internal country code → real ISO-2 code (identity when already ISO)."""
    return COUNTRY_CODE_TO_ISO.get(code, code)


def _iso_country_sql(column: str = "country") -> str:
    """`to_iso_country()` as a SQL expression, for the raw-SQL query paths.

    The codes are interpolated, not bound — COUNTRY_CODE_TO_ISO must stay a
    hardcoded module constant and must NEVER be fed runtime/user input.
    """
    whens = " ".join(f"WHEN '{internal}' THEN '{iso}'" for internal, iso in COUNTRY_CODE_TO_ISO.items())
    return f"CASE {column} {whens} ELSE {column} END"


# `local_incident_filter()` as a raw-SQL predicate, for the text() queries in
# fmedia/embeddings.py and fmedia/search.py that cannot use the ORM expression.
LOCAL_INCIDENT_SQL = (
    f"(incident_country IS NULL OR incident_country IN (country, {_iso_country_sql()}))"
)


def local_incident_filter():
    """True when an article counts as a LOCAL relevant incident for its own `country`.

    incident_country IS NULL  → not determined by the LLM (or by the classifier
                                 running before this feature existed) — treated
                                 as local, same as before this feature shipped.
    incident_country == country → LLM confirmed it happened in the monitored
                                 outlet's own country.
    incident_country == ISO(country) → same, but for a monitored country whose
                                 internal code is not real ISO-2: a UK outlet
                                 has `country` "UK" while the classifier writes
                                 the true ISO code "GB" for a UK incident, so
                                 `country` is also compared after normalization
                                 through COUNTRY_CODE_TO_ISO. Both spellings
                                 count as local (rows predating the classifier's
                                 ISO codes literally store "UK").
    Anything else (a different ISO-2 code) is a foreign incident: still a real,
    `confirmed` relevant incident article, but excluded wherever this filter is applied.
    """
    iso_country = case(COUNTRY_CODE_TO_ISO, value=Article.country, else_=Article.country)
    return or_(
        Article.incident_country.is_(None),
        Article.incident_country == Article.country,
        Article.incident_country == iso_country,
    )


# ── Case model — one Case = one real-world relevant incident incident ─────────────────

class Case(Base):
    __tablename__ = "cases"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Victim
    victim_name         = Column(String, nullable=True, index=True)
    victim_age          = Column(Integer, nullable=True)
    victim_nationality  = Column(String, nullable=True)

    # Incident
    country             = Column(String(2), nullable=False, index=True)
    incident_location   = Column(String, nullable=True)
    incident_region     = Column(String, nullable=True)
    incident_date       = Column(Date, nullable=True, index=True)

    # Classification
    event_type       = Column(String, nullable=True)
    event_relation   = Column(String, nullable=True)
    event_motive     = Column(String, nullable=True)

    # Perpetrator
    perpetrator_name    = Column(String, nullable=True)
    perpetrator_age     = Column(Integer, nullable=True)
    perpetrator_status  = Column(String, nullable=True)

    # Context
    had_prior_complaints = Column(Boolean, nullable=True)
    children_orphaned    = Column(Integer, nullable=True)
    weapon               = Column(String, nullable=True)

    # Summary
    summary             = Column(Text, nullable=True)

    # Metadata
    article_count       = Column(Integer, default=1)
    first_reported_at   = Column(DateTime, nullable=True)
    created_at          = Column(DateTime, server_default=func.now())
    updated_at          = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # : tombstone. Set when this case was merged into another one by
    # fmedia/case_merge_apply.py; its articles were moved to `merged_into_id`.
    # Listings and the coverage numerator skip tombstones; `resolve_case_id()`
    # follows the chain to the survivor. NULL for every live case.
    merged_into_id      = Column(Integer, ForeignKey("cases.id"), nullable=True, index=True)

    # Relationship
    articles = relationship("Article", backref="case", foreign_keys="Article.case_id")

    def __repr__(self):
        return (
            f"<Case #{self.id} {self.victim_name or 'unnamed'} "
            f"({self.country}) {self.incident_date} — {self.event_type}>"
        )


# ── Official counts ───────────────────
# Optional aggregate reference figures (one row per country/year) that an
# operator can load to compare coverage against an official total. Not used by
# the ingestion pipeline itself.
class OfficialCount(Base):
    __tablename__ = "official_counts"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    country     = Column(String(2), nullable=False, index=True)
    year        = Column(Integer, nullable=False, index=True)
    count       = Column(Integer, nullable=False)
    source_name = Column(String(200), nullable=True)
    source_url  = Column(String(500), nullable=True)
    notes       = Column(Text, nullable=True)

    def __repr__(self):
        return f"<OfficialCount {self.country} {self.year}: {self.count}>"


class StageDailyCount(Base):
    """Daily per-country pipeline stage counts. One row per
    (day, country); recomputed from existing Article columns, never LLM-derived.
    See fmedia/stage_metrics.py for each stage's definition."""
    __tablename__ = "stage_daily_counts"
    __table_args__ = (UniqueConstraint("day", "country", name="uq_stage_daily_day_country"),)

    id               = Column(Integer, primary_key=True, autoincrement=True)
    day              = Column(Date, nullable=False, index=True)
    country          = Column(String(2), nullable=False, index=True)
    fetched          = Column(Integer, nullable=False, default=0)
    kept             = Column(Integer, nullable=False, default=0)
    confirmed        = Column(Integer, nullable=False, default=0)
    enriched         = Column(Integer, nullable=False, default=0)
    with_victim_name = Column(Integer, nullable=False, default=0)
    embedded         = Column(Integer, nullable=False, default=0)
    computed_at      = Column(DateTime, default=datetime.utcnow)


# ── Case Dossier (Expediente de Caso) ─────────────────────────────────────────
# Curated editorial layer documenting how media coverage of ONE case evolved.
# Ethical guardrails are enforced at the model level:
#   * victim name renders ONLY if victim_name_public is True
#   * images render ONLY if image_rights_verified is True (display_image property)
#   * every factual claim carries an epistemic status (hecho/alegación/versión de medio)
#   * NO fields exist that could identify minors related to the case
#   * NO perpetrator-name field exists — presumption of innocence by design

JUDICIAL_STATUSES = {"investigacion", "juicio_en_curso", "sentencia_firme"}
DOSSIER_STATUSES  = {"en_seguimiento", "cobertura_cerrada", "en_espera_de_sentencia"}
FACT_STATUSES     = {"hecho_judicial", "alegacion_fiscalia", "version_medio_sin_confirmar"}
COVERAGE_FOCUS    = {"victima", "agresor", "mixto"}
SHORT_QUOTE_MAX   = 300  # chars — short quote with attribution, never long verbatim excerpts

_slug_re = _re_mod.compile(r"[^a-z0-9]+")


def make_dossier_slug(incident_date, city: str) -> str:
    """Internal identifier: date + city, never the victim's name."""
    c = _unicodedata.normalize("NFD", city or "")
    c = "".join(ch for ch in c if _unicodedata.category(ch) != "Mn").lower()
    c = _slug_re.sub("-", c).strip("-") or "sin-ciudad"
    d = str(incident_date)[:7] if incident_date else "sin-fecha"  # YYYY-MM
    return f"{d}-{c}"


class Dossier(Base):
    __tablename__ = "dossiers"

    id      = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(Integer, ForeignKey("cases.id"), nullable=True)  # optional link to pipeline Case

    slug          = Column(String(120), unique=True, nullable=False, index=True)
    country       = Column(String(2), nullable=False, index=True)
    city          = Column(String(120), nullable=False)
    region        = Column(String(120), nullable=True)
    incident_date = Column(Date, nullable=False)

    victim_name        = Column(String, nullable=True)
    victim_name_public = Column(Boolean, default=False, nullable=False)
    victim_age         = Column(Integer, nullable=True)
    victim_age_public  = Column(Boolean, default=False, nullable=False)

    # `city`/`region` stay populated for internal use (slug generation, case
    # linking) regardless of this flag — only PUBLIC RENDERING is gated, via
    # display_location below. Age + city + region together are often enough
    # to re-identify a victim even with the name withheld.
    location_public    = Column(Boolean, default=False, nullable=False)

    judicial_status = Column(String(30), default="investigacion", nullable=False)
    dossier_status  = Column(String(30), default="en_seguimiento", nullable=False)

    framing_analysis = Column(Text, nullable=True)

    image_url             = Column(Text, nullable=True)
    image_rights_verified = Column(Boolean, default=False, nullable=False)

    is_test_case = Column(Boolean, default=False, nullable=False)
    published    = Column(Boolean, default=False, nullable=False)

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    entries = relationship("DossierEntry", backref="dossier",
                           order_by="DossierEntry.entry_date",
                           cascade="all, delete-orphan")
    facts   = relationship("DossierFact", backref="dossier",
                           cascade="all, delete-orphan")

    @validates("judicial_status")
    def _v_judicial(self, key, value):
        if value not in JUDICIAL_STATUSES:
            raise ValueError(f"judicial_status inválido: {value!r} (válidos: {sorted(JUDICIAL_STATUSES)})")
        return value

    @validates("dossier_status")
    def _v_status(self, key, value):
        if value not in DOSSIER_STATUSES:
            raise ValueError(f"dossier_status inválido: {value!r} (válidos: {sorted(DOSSIER_STATUSES)})")
        return value

    @property
    def display_name(self):
        """Victim name for rendering — None unless explicitly public."""
        return self.victim_name if self.victim_name_public else None

    @property
    def display_age(self):
        """Victim age for rendering — None unless explicitly public."""
        return self.victim_age if self.victim_age_public else None

    @property
    def display_city(self):
        """City for rendering — None unless location is explicitly public."""
        return self.city if self.location_public else None

    @property
    def display_region(self):
        """Region for rendering — None unless location is explicitly public."""
        return self.region if self.location_public else None

    @property
    def display_image(self):
        """Image URL for rendering — None unless rights are verified. GUARDRAIL."""
        return self.image_url if self.image_rights_verified else None

    def __repr__(self):
        return f"<Dossier #{self.id} {self.slug} ({self.country}) {self.judicial_status}>"


class DossierEntry(Base):
    """One coverage-timeline entry: an article/segment about the case, with framing flags."""
    __tablename__ = "dossier_entries"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    dossier_id = Column(Integer, ForeignKey("dossiers.id"), nullable=False, index=True)

    entry_date          = Column(Date, nullable=False)
    outlet              = Column(String(200), nullable=False)
    headline_paraphrase = Column(Text, nullable=False)  # reformulated headline, never long verbatim
    source_url          = Column(Text, nullable=True)

    # Framing flags
    uses_topic_term = Column(Boolean, nullable=True)  # None = not assessed
    uses_euphemism     = Column(Boolean, nullable=True)
    coverage_focus     = Column(String(20), nullable=True)  # victima | agresor | mixto

    # Short quote WITH attribution — both or neither
    short_quote       = Column(String(SHORT_QUOTE_MAX), nullable=True)
    quote_attribution = Column(String(200), nullable=True)

    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        # GUARDRAIL at schema level: a quote may never exist without attribution
        CheckConstraint(
            "(short_quote IS NULL) OR (quote_attribution IS NOT NULL)",
            name="ck_entry_quote_requires_attribution",
        ),
    )

    @validates("coverage_focus")
    def _v_focus(self, key, value):
        if value is not None and value not in COVERAGE_FOCUS:
            raise ValueError(f"coverage_focus inválido: {value!r} (válidos: {sorted(COVERAGE_FOCUS)})")
        return value

    @validates("short_quote")
    def _v_quote(self, key, value):
        if value is not None and len(value) > SHORT_QUOTE_MAX:
            raise ValueError(f"short_quote supera {SHORT_QUOTE_MAX} caracteres — usar paráfrasis")
        return value


class DossierFact(Base):
    """A single factual claim about the case, ALWAYS tagged with epistemic status."""
    __tablename__ = "dossier_facts"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    dossier_id = Column(Integer, ForeignKey("dossiers.id"), nullable=False, index=True)

    fact_text   = Column(Text, nullable=False)
    fact_status = Column(String(40), nullable=False)  # hecho_judicial | alegacion_fiscalia | version_medio_sin_confirmar

    created_at = Column(DateTime, server_default=func.now())

    @validates("fact_status")
    def _v_fact_status(self, key, value):
        if value not in FACT_STATUSES:
            raise ValueError(f"fact_status inválido: {value!r} (válidos: {sorted(FACT_STATUSES)})")
        return value


# ── Backfill progress — persisted in DB so redeploys don't lose state ──

class BackfillProgress(Base):
    __tablename__ = "backfill_progress"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    country    = Column(String(5),  nullable=False)
    source     = Column(String(30), nullable=False)
    chunk_key  = Column(String(20), nullable=False)   # "2000-01" or "last-90d"
    status     = Column(String(20), default="pending") # pending/running/done/error
    fetched    = Column(Integer, default=0)
    saved      = Column(Integer, default=0)
    error_msg  = Column(Text, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("country", "source", "chunk_key", name="uq_bf_chunk"),
    )


# ── App-wide key-value settings ───────────────────────────────────────────────

class AppSetting(Base):
    __tablename__ = "app_settings"

    key   = Column(String(100), primary_key=True)
    value = Column(Text, nullable=True)


# ── Case-linking agent audit trail ──
# Fully ORM-mapped (unlike the embedding column) — no dialect-specific raw SQL
# needed, so Base.metadata.create_all() creates it identically on SQLite and
# Postgres. alembic/versions/002 also creates it (idempotently) for environments
# that run `alembic upgrade head` without ever calling init_db()/create_all().
# One row per case_agent.run_case_agent() invocation — persisted in BOTH shadow
# and active mode (shadow just never mutates Case/Article), which is what makes
# GET /admin/agent-decisions useful for deciding the shadow→active flip.

CASE_LINK_DECISIONS = {"merge", "new_case", "escalate"}
CASE_LINK_MODES = {"shadow", "active"}


class CaseLinkAudit(Base):
    __tablename__ = "case_link_audit"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    article_id        = Column(Integer, ForeignKey("articles.id"), nullable=False, index=True)
    decision          = Column(String(20), nullable=False)   # merge | new_case | escalate
    matched_case_id   = Column(Integer, ForeignKey("cases.id"), nullable=True)
    agent_reasoning   = Column(Text, nullable=True)
    difflib_decision  = Column(String(20), nullable=True)
    difflib_ratio     = Column(Float, nullable=True)
    iterations        = Column(Integer, nullable=False, default=0)
    tokens_in         = Column(Integer, nullable=False, default=0)
    tokens_out        = Column(Integer, nullable=False, default=0)
    cost_usd          = Column(Float, nullable=False, default=0.0)
    model_used        = Column(String(100), nullable=True)
    mode              = Column(String(10), nullable=False, default="shadow")  # shadow | active
    created_at        = Column(DateTime, default=datetime.utcnow)

    @validates("decision")
    def _v_decision(self, key, value):
        if value not in CASE_LINK_DECISIONS:
            raise ValueError(f"decision inválida: {value!r} (válidas: {sorted(CASE_LINK_DECISIONS)})")
        return value

    @validates("mode")
    def _v_mode(self, key, value):
        if value not in CASE_LINK_MODES:
            raise ValueError(f"mode inválido: {value!r} (válidos: {sorted(CASE_LINK_MODES)})")
        return value

    def __repr__(self):
        return (
            f"<CaseLinkAudit article={self.article_id} decision={self.decision} "
            f"case={self.matched_case_id} mode={self.mode}>"
        )


class CaseMergeAudit(Base):
    """One row per ABSORBED case of a merge — everything needed to
    undo it: which articles moved, what the survivor looked like before, and a
    snapshot of the absorbed case. `unmerge_batch()` reads these in reverse."""
    __tablename__ = "case_merge_audit"

    id                 = Column(Integer, primary_key=True, autoincrement=True)
    batch_id           = Column(String(64), nullable=False, index=True)
    tier               = Column(String(1), nullable=False)             # A | B | C
    survivor_id        = Column(Integer, ForeignKey("cases.id"), nullable=False, index=True)
    absorbed_id        = Column(Integer, ForeignKey("cases.id"), nullable=False, index=True)
    absorbed_snapshot  = Column(Text, nullable=False)                   # JSON of the absorbed Case row
    survivor_before    = Column(Text, nullable=False)                   # JSON of survivor fields we changed
    moved_article_ids  = Column(Text, nullable=False, default="[]")     # JSON list[int]
    reasons            = Column(Text, nullable=True)
    created_at         = Column(DateTime, default=datetime.utcnow, nullable=False)
    reverted_at        = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<CaseMergeAudit batch={self.batch_id} {self.absorbed_id}->{self.survivor_id} tier={self.tier}>"


class ContactMessage(Base):
    __tablename__ = "contact_messages"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    name       = Column(String(200), nullable=False)
    email      = Column(String(320), nullable=False)
    subject    = Column(String(300), nullable=True)
    message    = Column(Text, nullable=False)
    read       = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self):
        return f"<ContactMessage #{self.id} from={self.email} read={self.read}>"


def _migrate_add_typology_columns():
    """Add event_type/relacion/motivo + victim extraction + case columns if missing."""
    cols = [
        # Typology
        ("event_type",     "VARCHAR(50)"),
        ("event_relation", "VARCHAR(50)"),
        ("event_motive",   "VARCHAR(50)"),
        ("ml_score",          "FLOAT"),
        # Victim extraction
        ("victim_name",              "VARCHAR"),
        ("victim_age",               "INTEGER"),
        ("victim_nationality",       "VARCHAR"),
        ("incident_location",        "VARCHAR"),
        ("incident_region",          "VARCHAR"),
        ("incident_date_extracted",  "DATE"),
        ("perpetrator_name",         "VARCHAR"),
        ("perpetrator_age",          "INTEGER"),
        ("perpetrator_status",       "VARCHAR"),
        ("had_prior_complaints",     "BOOLEAN"),
        ("children_orphaned",        "INTEGER"),
        ("weapon",                   "VARCHAR"),
        ("event_summary",         "TEXT"),
        # Case grouping
        ("case_id",                  "INTEGER"),
        # LLM classification timestamp
        ("llm_classified_at",        "TIMESTAMP"),
        # Origin tracking
        ("origin",                   "VARCHAR(20) DEFAULT 'daily'"),
        # LLM cost tracking
        ("llm_model_used",           "VARCHAR(100)"),
        ("llm_tokens_in",            "INTEGER"),
        ("llm_tokens_out",           "INTEGER"),
        ("llm_cost_usd",             "FLOAT"),
        # RAG embeddings — SQLite/JSON backing column, see Article model docstring
        ("embedding_json",           "TEXT"),
        # Incident country — see Article model docstring
        ("incident_country",         "VARCHAR(2)"),
        # Full scraped article body — see Article model docstring
        ("full_text",                "TEXT"),
        # Enrichment attempt tracking
        ("enrichment_attempts",        "INTEGER DEFAULT 0"),
        ("enrichment_last_attempt_at", "TIMESTAMP"),
        ("enrichment_last_result",     "VARCHAR(40)"),
    ]
    is_pg = DB_PATH.startswith("postgresql")
    # ADD COLUMN IF NOT EXISTS needs an ACCESS EXCLUSIVE lock even when the column exists, so one
    # open transaction on `articles` used to hang every start-up. Only touch what is missing.
    existing = {c["name"] for c in inspect(engine).get_columns("articles")}
    missing = [(name, ctype) for name, ctype in cols if name not in existing]
    if not missing:
        return
    with engine.connect() as conn:
        if is_pg:
            conn.execute(text("SET lock_timeout = '10s'"))
        for col_name, col_type in missing:
            try:
                clause = "ADD COLUMN IF NOT EXISTS" if is_pg else "ADD COLUMN"
                conn.execute(text(f"ALTER TABLE articles {clause} {col_name} {col_type}"))
                conn.commit()
            except Exception as e:
                conn.rollback()
                err = str(e).lower()
                if "already exists" not in err and "duplicate column" not in err:
                    print(f"[db] Migration warning for {col_name}: {e}")


def _backfill_types_from_section():
    """Move relevant incident type from section column to event_type for already-classified articles."""
    db = SessionLocal()
    try:
        arts = db.query(Article).filter(
            Article.event_type.is_(None),
            Article.review_status == "confirmed",
            Article.section.in_(list(POSITIVE_TYPE_IDS)),
        ).all()
        if not arts:
            return
        for art in arts:
            art.event_type = art.section
            art.section = None
        db.commit()
        print(f"[db] Backfilled event_type for {len(arts)} articles from section column.")
    except Exception as e:
        db.rollback()
        print(f"[db] Backfill error: {e}")
    finally:
        db.close()


def _migrate_add_fingerprint_column():
    """Add content_fingerprint column if it doesn't exist yet."""
    insp = inspect(engine)
    has_column = "content_fingerprint" in {c["name"] for c in insp.get_columns("articles")}
    has_index = "ix_articles_content_fingerprint" in {i["name"] for i in insp.get_indexes("articles")}
    if has_column and has_index:
        return  # nothing to do: avoid taking table locks at start-up
    with engine.connect() as conn:
        try:
            if DB_PATH.startswith("postgresql"):
                conn.execute(text("SET lock_timeout = '10s'"))
            conn.execute(text(
                "ALTER TABLE articles ADD COLUMN IF NOT EXISTS content_fingerprint VARCHAR(64)"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_articles_content_fingerprint ON articles (content_fingerprint)"
            ))
            conn.commit()
        except Exception:
            conn.rollback()


def _migrate_add_case_merge_column():
    """cases.merged_into_id. Same lock-free pattern as the articles
    migrations: inspect first, only ALTER when the column is really missing.
    alembic/versions/011 is the canonical migration; this keeps SQLite/dev DBs
    created before #29 bootable."""
    existing = {c["name"] for c in inspect(engine).get_columns("cases")}
    if "merged_into_id" in existing:
        return
    is_pg = DB_PATH.startswith("postgresql")
    with engine.connect() as conn:
        try:
            if is_pg:
                conn.execute(text("SET lock_timeout = '10s'"))
                conn.execute(text("ALTER TABLE cases ADD COLUMN IF NOT EXISTS merged_into_id INTEGER"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS ix_cases_merged_into_id ON cases (merged_into_id)"))
            else:
                conn.execute(text("ALTER TABLE cases ADD COLUMN merged_into_id INTEGER"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS ix_cases_merged_into_id ON cases (merged_into_id)"))
            conn.commit()
        except Exception as e:
            conn.rollback()
            if "already exists" not in str(e).lower() and "duplicate column" not in str(e).lower():
                print(f"[db] Migration warning for cases.merged_into_id: {e}")


def init_db():
    os.makedirs("data", exist_ok=True)
    Base.metadata.create_all(bind=engine)
    _migrate_add_typology_columns()
    _backfill_types_from_section()
    _migrate_add_fingerprint_column()
    _migrate_add_case_merge_column()
    # BackfillProgress and AppSetting are created by create_all above (new tables)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Write ──────────────────────────────────────────────────────────────────────

def _content_fingerprint(title: str, country: str) -> str:
    import hashlib, re
    normalized = re.sub(r"\s+", " ", (title or "").lower().strip())
    return hashlib.sha256(f"{country}:{normalized}".encode()).hexdigest()[:32]


# Canonical source names — RSS feeds and scrapers write inconsistent variants
# of the same outlet (e.g. a bare hostname vs. the display name), which splits
# per-source aggregates in two. Normalize at write time so new rows don't add
# to the split; existing rows are left as-is (no backfill migration here).
SOURCE_NAME_MAP = {
    "clarin.com":            "Clarín",
    "milenio":                "MILENIO",
    "cooperativa.cl":         "Cooperativa",
    "animal politico":        "Animal Político",
}


def _canonical_source(name: Optional[str]) -> Optional[str]:
    if not name:
        return name
    return SOURCE_NAME_MAP.get(name.strip().lower(), name)


def save_article(db: Session, article: dict) -> Optional[Article]:
    try:
        if db.query(Article).filter_by(guardian_id=article["guardian_id"]).first():
            return None
        title = article.get("title", "")
        country = article.get("country", "")
        if article.get("source"):
            article = {**article, "source": _canonical_source(article["source"])}
        # Content fingerprint dedup — catches cross-source duplicates with same title
        fp = _content_fingerprint(title, country)
        if title and db.query(Article).filter_by(content_fingerprint=fp).first():
            return None
        data = {k: v for k, v in article.items() if hasattr(Article, k)}
        data["content_fingerprint"] = fp
        # Convert published_date string → Python date object
        pd_val = data.get("published_date")
        if isinstance(pd_val, str):
            try:
                data["published_date"] = datetime.strptime(pd_val[:10], "%Y-%m-%d").date()
            except Exception:
                data["published_date"] = date.today()
        row = Article(**data)
        if "review_status" not in data:
            row.review_status = "pending"
        db.add(row)
        db.commit()
        db.refresh(row)
        return row
    except Exception as e:
        db.rollback()
        print(f"[db] save_article failed: {e} — {article.get('title', '')[:50]}")
        return None


def review_article(db: Session, article_id: int, decision: str, reviewer: str) -> Optional[Article]:
    """decision: 'confirm' or 'reject'"""
    row = db.query(Article).filter_by(id=article_id).first()
    if not row:
        return None
    row.review_status = "confirmed" if decision == "confirm" else "rejected"
    row.human_label   = 1 if decision == "confirm" else 0
    row.reviewed_by   = reviewer
    row.reviewed_at   = datetime.utcnow()
    db.commit()
    db.refresh(row)
    return row


# ── Training data ──────────────────────────────────────────────────────────────

def get_labeled_articles(db: Session, country: Optional[str] = None) -> List[Article]:
    q = db.query(Article).filter(Article.human_label.isnot(None))
    if country:
        q = q.filter(Article.country == country)
    return q.all()


def count_new_labels(db: Session, since: Optional[datetime]) -> int:
    q = db.query(Article).filter(Article.human_label.isnot(None))
    if since:
        q = q.filter(Article.reviewed_at > since)
    return q.count()


# ── Public reads (confirmed only) ──────────────────────────────────────────────

def get_articles_by_date(db: Session, day: date, country: Optional[str] = None) -> List[Article]:
    q = db.query(Article).filter(
        Article.published_date == day, Article.review_status == "confirmed",
        local_incident_filter(),
    )
    if country:
        q = q.filter(Article.country == country)
    return q.order_by(desc(Article.published_date)).all()


def get_articles_paginated(
    db: Session, page: int = 1, per_page: int = 20,
    year: Optional[int] = None, month: Optional[int] = None,
    status: str = "confirmed", country: Optional[str] = None,
    event_type: Optional[str] = None,
    event_relation: Optional[str] = None,
    event_motive: Optional[str] = None,
    search: Optional[str] = None,
) -> dict:
    q = db.query(Article).filter(Article.review_status == status)
    if status == "confirmed":
        q = q.filter(local_incident_filter())
    if year:
        q = q.filter(extract("year", Article.published_date) == year)
    if month:
        q = q.filter(extract("month", Article.published_date) == month)
    if country:
        q = q.filter(Article.country == country)
    if event_type:
        q = q.filter(Article.event_type == event_type)
    if event_relation:
        q = q.filter(Article.event_relation == event_relation)
    if event_motive:
        q = q.filter(Article.event_motive == event_motive)
    if search:
        like = f"%{search.strip()}%"
        q = q.filter(or_(Article.title.ilike(like), Article.content_snippet.ilike(like)))
    total = q.count()
    items = q.order_by(desc(Article.published_date)).offset((page-1)*per_page).limit(per_page).all()
    return {"total": total, "page": page, "per_page": per_page, "items": items}


def get_monthly_counts(db: Session, status: str = "confirmed", country: Optional[str] = None) -> List[dict]:
    q = db.query(
        cast(extract("year", Article.published_date), String).label("year"),
        cast(extract("month", Article.published_date), String).label("month"),
        func.count().label("count"),
    ).filter(Article.review_status == status)
    if status == "confirmed":
        q = q.filter(local_incident_filter())
    if country:
        q = q.filter(Article.country == country)
    rows = q.group_by("year", "month").order_by("year", "month").all()
    return [{"year": int(r.year), "month": int(r.month), "count": r.count} for r in rows]


def get_total_count(db: Session, status: str = "confirmed", country: Optional[str] = None) -> int:
    q = db.query(Article).filter(Article.review_status == status)
    if status == "confirmed":
        q = q.filter(local_incident_filter())
    if country:
        q = q.filter(Article.country == country)
    return q.count()


def get_date_range(db: Session, status: str = "confirmed", country: Optional[str] = None) -> dict:
    q = db.query(Article).filter(Article.review_status == status)
    if status == "confirmed":
        q = q.filter(local_incident_filter())
    if country:
        q = q.filter(Article.country == country)
    first = q.with_entities(func.min(Article.published_date)).scalar()
    last  = q.with_entities(func.max(Article.published_date)).scalar()
    return {"first": str(first) if first else None, "last": str(last) if last else None}


def get_queue_stats(db: Session, country: Optional[str] = None) -> dict:
    def _count(status):
        q = db.query(Article).filter(Article.review_status == status)
        if status == "confirmed":
            q = q.filter(local_incident_filter())
        if country:
            q = q.filter(Article.country == country)
        return q.count()
    labeled_q = db.query(Article).filter(Article.human_label.isnot(None))
    if country:
        labeled_q = labeled_q.filter(Article.country == country)
    return {
        "pending":   _count("pending"),
        "confirmed": _count("confirmed"),
        "rejected":  _count("rejected"),
        "labeled":   labeled_q.count(),
    }


# ── Re-classification support ─────────────────────────────────────────────────

def get_article_by_id(db: Session, article_id: int) -> Optional[Article]:
    return db.query(Article).filter_by(id=article_id).first()


def get_articles_by_status(
    db: Session,
    status: str,
    page: int = 1,
    per_page: int = 20,
    country: Optional[str] = None,
) -> dict:
    """Get articles by any status for the stats panel."""
    q = db.query(Article).filter(Article.review_status == status)
    if status == "confirmed":
        q = q.filter(local_incident_filter())
    if country:
        q = q.filter(Article.country == country)
    total = q.count()
    items = q.order_by(desc(Article.published_date)).offset((page-1)*per_page).limit(per_page).all()
    return {"total": total, "page": page, "per_page": per_page, "items": items}


def get_training_articles(db: Session, page: int = 1, per_page: int = 20) -> dict:
    """All articles that have a human label (both confirmed and rejected)."""
    q     = db.query(Article).filter(Article.human_label.isnot(None))
    total = q.count()
    items = q.order_by(desc(Article.reviewed_at)).offset((page-1)*per_page).limit(per_page).all()
    return {"total": total, "page": page, "per_page": per_page, "items": items}


# ── Training history ───────────────────────────────────────────────────────────

class TrainingRun(Base):
    __tablename__ = "training_runs"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    trained_at  = Column(DateTime, default=datetime.utcnow)
    n_samples   = Column(Integer)
    n_positive  = Column(Integer)
    n_negative  = Column(Integer)
    accuracy    = Column(String(10))
    precision   = Column(String(10))
    recall      = Column(String(10))
    f1          = Column(String(10))
    source      = Column(String(100))  # "csv", "human_labels", "combined"
    notes       = Column(Text, nullable=True)


def save_training_run(db: Session, metrics: dict, source: str = "combined", notes: str = None):
    run = TrainingRun(
        n_samples  = metrics.get("n_samples", 0),
        n_positive = metrics.get("n_positive", 0),
        n_negative = metrics.get("n_negative", 0),
        accuracy   = str(metrics.get("accuracy", "")),
        precision  = str(metrics.get("precision", "")),
        recall     = str(metrics.get("recall", "")),
        f1         = str(metrics.get("f1", "")),
        source     = source,
        notes      = notes,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def get_training_history(db: Session) -> list:
    return db.query(TrainingRun).order_by(desc(TrainingRun.trained_at)).all()


# ── Typology queries ──────────────────────────────────────────────────────────

def get_type_counts(db: Session, country: Optional[str] = None) -> dict:
    """Return {type: count} for confirmed articles with a event_type."""
    q = db.query(
        Article.event_type, func.count().label("count")
    ).filter(
        Article.review_status == "confirmed",
        Article.event_type.isnot(None),
        local_incident_filter(),
    )
    if country:
        q = q.filter(Article.country == country)
    rows = q.group_by(Article.event_type).all()
    return {r.event_type: r.count for r in rows}


def get_relacion_counts(db: Session, country: Optional[str] = None) -> dict:
    q = db.query(
        Article.event_relation, func.count().label("count")
    ).filter(
        Article.review_status == "confirmed",
        Article.event_relation.isnot(None),
        local_incident_filter(),
    )
    if country:
        q = q.filter(Article.country == country)
    rows = q.group_by(Article.event_relation).all()
    return {r.event_relation: r.count for r in rows}


def get_motivo_counts(db: Session, country: Optional[str] = None) -> dict:
    q = db.query(
        Article.event_motive, func.count().label("count")
    ).filter(
        Article.review_status == "confirmed",
        Article.event_motive.isnot(None),
        local_incident_filter(),
    )
    if country:
        q = q.filter(Article.country == country)
    rows = q.group_by(Article.event_motive).all()
    return {r.event_motive: r.count for r in rows}


def get_type_counts_by_month(db: Session, country: Optional[str] = None) -> list:
    """Return [{year, month, type, count}] for stacked area chart."""
    q = db.query(
        cast(extract("year", Article.published_date), String).label("year"),
        cast(extract("month", Article.published_date), String).label("month"),
        Article.event_type,
        func.count().label("count"),
    ).filter(
        Article.review_status == "confirmed",
        Article.event_type.isnot(None),
        local_incident_filter(),
    )
    if country:
        q = q.filter(Article.country == country)
    rows = q.group_by("year", "month", Article.event_type).order_by("year", "month").all()
    return [
        {"year": int(r.year), "month": int(r.month), "type": r.event_type, "count": r.count}
        for r in rows
    ]


def get_cross_tabulation(db: Session, country: str = None) -> list:
    """Return [{country, type, relacion, count}] for Sankey diagram."""
    q = db.query(
        Article.country,
        Article.event_type,
        Article.event_relation,
        func.count().label("count"),
    ).filter(
        Article.review_status == "confirmed",
        Article.event_type.isnot(None),
        local_incident_filter(),
    )
    if country:
        q = q.filter(Article.country == country.upper())
    rows = q.group_by(Article.country, Article.event_type, Article.event_relation).all()
    return [
        {"country": r.country, "type": r.event_type, "relacion": r.event_relation or "no_aplica", "count": r.count}
        for r in rows
    ]
