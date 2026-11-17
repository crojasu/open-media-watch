"""
dossiers.py — Expediente de Caso (Case Dossier)
------------------------------------------------
Curated editorial layer: one dossier per case, documenting how media coverage
of that case evolved over time (timeline of entries with framing flags), plus a
free-text framing analysis.

Public routes (published dossiers only):
  GET /api/dossiers                  → listing
  GET /api/dossiers/by-slug/{slug}   → detail

Write API (HTTP Basic — reviewer credentials, same as review panel):
  POST  /api/dossiers                → create dossier
  PATCH /api/dossiers/{id}           → update fields (judicial status, analysis, …)
  POST  /api/dossiers/{id}/entries   → add coverage-timeline entry
  POST  /api/dossiers/{id}/facts     → add epistemically-tagged fact
  POST  /api/dossiers/{id}/publish   → make public

Ethical guardrails (see database.py Dossier model):
  * victim name renders only if victim_name_public
  * victim age renders only if victim_age_public (Dossier.display_age)
  * city/region render only if location_public (Dossier.display_city/display_region)
 — age + city + region together can re-identify a victim even without a name
  * image renders only if image_rights_verified (Dossier.display_image)
  * every fact carries hecho_judicial / alegacion_fiscalia / version_medio_sin_confirmar
  * no perpetrator-name field, no minor-related fields
"""
from __future__ import annotations

from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .auth import require_reviewer
from .database import (
    get_db, Dossier, DossierEntry, DossierFact, make_dossier_slug,
    JUDICIAL_STATUSES, DOSSIER_STATUSES, FACT_STATUSES, COVERAGE_FOCUS,
    SHORT_QUOTE_MAX,
)

router = APIRouter()

COUNTRY_NAMES = {"UK": "United Kingdom", "CL": "Chile", "AR": "Argentina",
                 "MX": "México", "ES": "España", "DE": "Deutschland"}
FLAGS = {"UK": "🇬🇧", "CL": "🇨🇱", "AR": "🇦🇷", "MX": "🇲🇽", "ES": "🇪🇸", "DE": "🇩🇪"}

JUDICIAL_LABELS = {
    "investigacion":   "Under investigation",
    "juicio_en_curso": "Trial in progress",
    "sentencia_firme": "Final sentence",
}
DOSSIER_STATUS_LABELS = {
    "en_seguimiento":         "Active monitoring",
    "cobertura_cerrada":      "Coverage closed",
    "en_espera_de_sentencia": "Awaiting sentence",
}
FACT_STATUS_LABELS = {
    "hecho_judicial":              "Confirmed judicial fact",
    "alegacion_fiscalia":          "Prosecution allegation",
    "version_medio_sin_confirmar": "Unconfirmed media account",
}
FOCUS_LABELS = {"victima": "Centers the victim", "agresor": "Centers the aggressor", "mixto": "Mixed"}


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class DossierCreate(BaseModel):
    country: str = Field(min_length=2, max_length=2)
    city: str = Field(min_length=1, max_length=120)
    incident_date: date
    region: Optional[str] = None
    case_id: Optional[int] = None
    victim_name: Optional[str] = None
    victim_name_public: bool = False
    victim_age: Optional[int] = Field(default=None, ge=0, le=130)
    victim_age_public: bool = False
    location_public: bool = False
    judicial_status: str = "investigacion"
    dossier_status: str = "en_seguimiento"
    framing_analysis: Optional[str] = None
    image_url: Optional[str] = None
    image_rights_verified: bool = False
    is_test_case: bool = False

    @field_validator("judicial_status")
    @classmethod
    def _judicial(cls, v):
        if v not in JUDICIAL_STATUSES:
            raise ValueError(f"judicial_status debe ser uno de {sorted(JUDICIAL_STATUSES)}")
        return v

    @field_validator("dossier_status")
    @classmethod
    def _status(cls, v):
        if v not in DOSSIER_STATUSES:
            raise ValueError(f"dossier_status debe ser uno de {sorted(DOSSIER_STATUSES)}")
        return v

    @field_validator("country")
    @classmethod
    def _country(cls, v):
        return v.upper()


class DossierPatch(BaseModel):
    judicial_status: Optional[str] = None
    dossier_status: Optional[str] = None
    framing_analysis: Optional[str] = None
    victim_name: Optional[str] = None
    victim_name_public: Optional[bool] = None
    victim_age: Optional[int] = Field(default=None, ge=0, le=130)
    victim_age_public: Optional[bool] = None
    location_public: Optional[bool] = None
    region: Optional[str] = None
    image_url: Optional[str] = None
    image_rights_verified: Optional[bool] = None
    case_id: Optional[int] = None

    @field_validator("judicial_status")
    @classmethod
    def _judicial(cls, v):
        if v is not None and v not in JUDICIAL_STATUSES:
            raise ValueError(f"judicial_status debe ser uno de {sorted(JUDICIAL_STATUSES)}")
        return v

    @field_validator("dossier_status")
    @classmethod
    def _status(cls, v):
        if v is not None and v not in DOSSIER_STATUSES:
            raise ValueError(f"dossier_status debe ser uno de {sorted(DOSSIER_STATUSES)}")
        return v


class EntryCreate(BaseModel):
    entry_date: date
    outlet: str = Field(min_length=1, max_length=200)
    headline_paraphrase: str = Field(min_length=1)
    source_url: Optional[str] = None
    uses_topic_term: Optional[bool] = None
    uses_euphemism: Optional[bool] = None
    coverage_focus: Optional[str] = None
    short_quote: Optional[str] = Field(default=None, max_length=SHORT_QUOTE_MAX)
    quote_attribution: Optional[str] = Field(default=None, max_length=200)

    @field_validator("coverage_focus")
    @classmethod
    def _focus(cls, v):
        if v is not None and v not in COVERAGE_FOCUS:
            raise ValueError(f"coverage_focus debe ser uno de {sorted(COVERAGE_FOCUS)}")
        return v

    @model_validator(mode="after")
    def _quote_needs_attribution(self):
        # GUARDRAIL: a quote may only appear as short quote WITH attribution.
        if self.short_quote and not self.quote_attribution:
            raise ValueError("short_quote requiere quote_attribution — nunca citas sin atribución")
        return self


class FactCreate(BaseModel):
    fact_text: str = Field(min_length=1)
    fact_status: str

    @field_validator("fact_status")
    @classmethod
    def _fs(cls, v):
        if v not in FACT_STATUSES:
            raise ValueError(f"fact_status debe ser uno de {sorted(FACT_STATUSES)}")
        return v


# ── Helpers ───────────────────────────────────────────────────────────────────

def has_published_dossiers(db: Session) -> bool:
    """Nav helper: the public 'Dossiers' link only shows once something is published."""
    return db.query(Dossier.id).filter(Dossier.published.is_(True)).first() is not None


def _unique_slug(db: Session, incident_date, city: str) -> str:
    base = make_dossier_slug(incident_date, city)
    slug, n = base, 2
    while db.query(Dossier).filter(Dossier.slug == slug).first():
        slug = f"{base}-{n}"
        n += 1
    return slug


def _dossier_or_404(db: Session, dossier_id: int) -> Dossier:
    d = db.get(Dossier, dossier_id)
    if not d:
        raise HTTPException(status_code=404, detail="Dossier not found")
    return d


def _serialize(d: Dossier) -> dict:
    return {
        "id": d.id, "slug": d.slug, "country": d.country, "city": d.city,
        "region": d.region, "incident_date": str(d.incident_date),
        "victim_name_public": d.victim_name_public,
        "victim_age": d.victim_age,
        "victim_age_public": d.victim_age_public,
        "location_public": d.location_public,
        "judicial_status": d.judicial_status, "dossier_status": d.dossier_status,
        "published": d.published, "is_test_case": d.is_test_case,
        "image_rights_verified": d.image_rights_verified,
        "entries": len(d.entries), "facts": len(d.facts),
    }


# ── Public read API (published dossiers only) ────────────────────────────────

@router.get("/api/dossiers")
def list_published_dossiers(db: Session = Depends(get_db), country: str = ""):
    q = db.query(Dossier).filter(Dossier.published.is_(True))
    country = country.upper()
    if country:  # a value matching no dossier simply returns an empty list
        q = q.filter(Dossier.country == country)
    return [{
        "slug": d.slug,
        "display_name": d.display_name,
        "victim_age": d.display_age,
        "country": d.country,
        "city": d.display_city,
        "region": d.display_region,
        "incident_date": str(d.incident_date),
        "judicial_status": d.judicial_status,
        "dossier_status": d.dossier_status,
        "is_test_case": d.is_test_case,
        "entry_count": len(d.entries),
    } for d in q.order_by(Dossier.incident_date.desc()).all()]


@router.get("/api/dossiers/by-slug/{slug}")
def get_published_dossier(slug: str, db: Session = Depends(get_db)):
    d = db.query(Dossier).filter(Dossier.slug == slug, Dossier.published.is_(True)).first()
    if not d:
        raise HTTPException(status_code=404, detail="Dossier not found")
    # GUARDRAIL: only explicitly-permitted scalars are returned, never the raw
    # ORM object, so victim_name / image_url are unreachable unless made public.
    return {
        "slug": d.slug,
        "incident_date": str(d.incident_date),
        "country": d.country,
        "city": d.display_city,
        "region": d.display_region,
        "victim_age": d.display_age,
        "display_name": d.display_name,
        "display_image": d.display_image,
        "judicial_status": d.judicial_status,
        "dossier_status": d.dossier_status,
        "is_test_case": d.is_test_case,
        "framing_analysis": d.framing_analysis,
        "entries": [{
            "entry_date": str(e.entry_date),
            "outlet": e.outlet,
            "headline_paraphrase": e.headline_paraphrase,
            "source_url": e.source_url,
            "uses_topic_term": e.uses_topic_term,
            "uses_euphemism": e.uses_euphemism,
            "coverage_focus": e.coverage_focus,
            "short_quote": e.short_quote,
            "quote_attribution": e.quote_attribution,
        } for e in d.entries],
        "facts": [{"text": f.fact_text, "status": f.fact_status} for f in d.facts],
    }


# ── Write API (reviewer auth) ─────────────────────────────────────────────────

@router.post("/api/dossiers", status_code=201)
def create_dossier(payload: DossierCreate, db: Session = Depends(get_db),
                   reviewer: str = Depends(require_reviewer)):
    d = Dossier(
        slug=_unique_slug(db, payload.incident_date, payload.city),
        **payload.model_dump(),
    )
    db.add(d)
    try:
        db.commit()
    except IntegrityError:
        # concurrent POSTs can race _unique_slug on the UNIQUE constraint
        db.rollback()
        raise HTTPException(status_code=409, detail="Slug collision — retry")
    db.refresh(d)
    return _serialize(d)


@router.patch("/api/dossiers/{dossier_id}")
def patch_dossier(dossier_id: int, payload: DossierPatch, db: Session = Depends(get_db),
                  reviewer: str = Depends(require_reviewer)):
    d = _dossier_or_404(db, dossier_id)
    for k, v in payload.model_dump(exclude_unset=True).items():
        setattr(d, k, v)
    db.commit()
    db.refresh(d)
    return _serialize(d)


# Nota editorial: se permite agregar entradas/hechos a expedientes publicados —
# el expediente es un documento vivo (la cobertura sigue evolucionando) y el rol
# reviewer es el mismo que publica. Si se introduce un flujo de re-aprobación,
# bloquear aquí cuando published=True.
@router.post("/api/dossiers/{dossier_id}/entries", status_code=201)
def add_entry(dossier_id: int, payload: EntryCreate, db: Session = Depends(get_db),
              reviewer: str = Depends(require_reviewer)):
    d = _dossier_or_404(db, dossier_id)
    e = DossierEntry(dossier_id=d.id, **payload.model_dump())
    db.add(e)
    db.commit()
    return {"id": e.id, "dossier_id": d.id}


@router.post("/api/dossiers/{dossier_id}/facts", status_code=201)
def add_fact(dossier_id: int, payload: FactCreate, db: Session = Depends(get_db),
             reviewer: str = Depends(require_reviewer)):
    d = _dossier_or_404(db, dossier_id)
    f = DossierFact(dossier_id=d.id, **payload.model_dump())
    db.add(f)
    db.commit()
    return {"id": f.id, "dossier_id": d.id}


@router.post("/api/dossiers/{dossier_id}/publish")
def publish_dossier(dossier_id: int, db: Session = Depends(get_db),
                    reviewer: str = Depends(require_reviewer)):
    d = _dossier_or_404(db, dossier_id)
    d.published = True
    db.commit()
    return _serialize(d)


# ── Bridge from the pipeline: bootstrap a dossier from an existing Case ───────

@router.post("/api/dossiers/from-case/{case_id}", status_code=201)
def create_dossier_from_case(case_id: int, db: Session = Depends(get_db),
                             reviewer: str = Depends(require_reviewer)):
    """
    Create a DRAFT dossier pre-filled from a pipeline Case.
    Guardrails: victim name is copied but NOT public by default; perpetrator
    and children data are never copied; the dossier stays unpublished.
    Timeline entries are NOT auto-created — headlines must be editorially
    paraphrased (see /suggested-entries).
    """
    from .database import Case
    case = db.get(Case, case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    existing = db.query(Dossier).filter(Dossier.case_id == case_id).first()
    if existing:
        raise HTTPException(status_code=409,
                            detail=f"Case {case_id} ya tiene expediente: {existing.slug}")

    incident_date = case.incident_date
    if incident_date is None and case.first_reported_at is not None:
        incident_date = case.first_reported_at.date()
    if incident_date is None:
        raise HTTPException(status_code=422,
                            detail="Case sin fecha (incident_date/first_reported_at) — completar antes")

    city = case.incident_location or "sin-ciudad"
    d = Dossier(
        slug=_unique_slug(db, incident_date, city),
        case_id=case.id,
        country=case.country,
        city=city,
        region=case.incident_region,
        incident_date=incident_date,
        victim_name=case.victim_name,
        victim_name_public=False,   # editorial decision required to flip
        victim_age=case.victim_age,
        victim_age_public=False,    # editorial decision required to flip
        location_public=False,      # editorial decision required to flip
        judicial_status="investigacion",
        dossier_status="en_seguimiento",
        is_test_case=False,
        published=False,
    )
    db.add(d)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="Slug collision — retry")
    db.refresh(d)
    return _serialize(d)


@router.get("/api/dossiers/{dossier_id}/suggested-entries")
def suggested_entries(dossier_id: int, db: Session = Depends(get_db),
                      reviewer: str = Depends(require_reviewer)):
    """
    Coverage the pipeline has linked to this dossier's Case that is not yet
    in the timeline (matched by source_url). Includes the original title
    ONLY for the reviewer to paraphrase — it never enters dossier tables
    nor any public page. Re-run anytime: new articles auto-linked to the
    Case appear here as fresh suggestions.
    """
    d = _dossier_or_404(db, dossier_id)
    if not d.case_id:
        return {"case_id": None, "pending": 0, "suggestions": [],
                "note": "Dossier sin Case vinculado — setear case_id via PATCH"}

    from .database import Article
    arts = (db.query(Article)
              .filter(Article.case_id == d.case_id,
                      Article.review_status == "confirmed")
              .order_by(Article.published_date.asc())
              .all())
    existing_urls = {e.source_url for e in d.entries if e.source_url}
    suggestions = [{
        "article_id": a.id,
        "published_date": str(a.published_date),
        "outlet": a.source,
        "source_url": a.web_url,
        "original_title": a.title,  # reviewer-only, to write headline_paraphrase
    } for a in arts if a.web_url not in existing_urls]

    return {"case_id": d.case_id, "pending": len(suggestions), "suggestions": suggestions}
