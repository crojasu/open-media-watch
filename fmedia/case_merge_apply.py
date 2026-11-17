"""
case_merge_apply.py — reversible application of case-merge proposals.

`merge_group()` folds the members of one `MergeGroup` into its survivor:
articles move to the survivor, the survivor is enriched from the absorbed
cases (nulls only), its country becomes the group's incident country, and each
absorbed case is tombstoned (`cases.merged_into_id`). Every absorbed case gets
a `case_merge_audit` row carrying the snapshot and the moved article ids, so
`unmerge_batch()` can put everything back exactly.

Nothing here commits: the caller (scripts/case_merge_apply.py) owns the
transaction so a batch is all-or-nothing.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import date, datetime
from typing import Dict, Iterable, List, Optional

from .case_merge import CaseRecord, MergeGroup, ref_date
from .database import Article, Case, CaseMergeAudit

SURVIVOR_FIELDS = (
    "victim_name", "victim_age", "victim_nationality", "country", "incident_location", "incident_region",
    "incident_date", "event_type", "event_relation", "event_motive", "perpetrator_name",
    "perpetrator_age", "perpetrator_status", "had_prior_complaints", "children_orphaned", "weapon",
    "summary", "article_count", "first_reported_at",
)
ENRICH_FIELDS = tuple(f for f in SURVIVOR_FIELDS if f not in ("country", "article_count", "first_reported_at", "summary"))


def _json_default(v):
    if isinstance(v, (date, datetime)):
        return v.isoformat()
    raise TypeError(str(type(v)))


def _snapshot(case: Case) -> Dict:
    return {f: getattr(case, f) for f in SURVIVOR_FIELDS}


def _restore(case: Case, snap: Dict) -> None:
    for f, v in snap.items():
        if f in ("incident_date",) and isinstance(v, str):
            v = date.fromisoformat(v)
        elif f == "first_reported_at" and isinstance(v, str):
            v = datetime.fromisoformat(v)
        setattr(case, f, v)


def _as_date(v) -> Optional[date]:
    if v is None:
        return None
    return v.date() if isinstance(v, datetime) else v


# ── loading ──────────────────────────────────────────────────────────────────

_CASE_COLS = ("id", "country", "victim_name", "victim_age", "incident_location", "incident_region", "incident_date",
              "first_reported_at", "event_type", "perpetrator_name", "article_count", "weapon", "summary")


def has_merge_column(db) -> bool:
    """True once cases.merged_into_id exists (alembic 011 / startup migration)."""
    from sqlalchemy import inspect
    bind = db.get_bind()
    return "merged_into_id" in {c["name"] for c in inspect(bind).get_columns("cases")}


def load_case_records(db, include_tombstones: bool = False) -> List[CaseRecord]:
    """CaseRecords for every live case, with their articles' incident countries,
    titles (<= 5) and earliest publication date. Plain SQL on explicit columns so
    the DRY-RUN works on a production DB that has not run migration 011 yet."""
    from sqlalchemy import text
    inc, titles, first_pub = {}, {}, {}
    rows = db.execute(text("SELECT case_id, incident_country, title, published_date FROM articles "
                           "WHERE case_id IS NOT NULL ORDER BY published_date, id"))
    for cid, icc, title, pub in rows:
        if icc:
            inc.setdefault(cid, []).append(icc)
        if title and len(titles.setdefault(cid, [])) < 5:
            titles[cid].append(title)
        d = _as_date(pub) if not isinstance(pub, str) else date.fromisoformat(pub[:10])
        if d and (cid not in first_pub or d < first_pub[cid]):
            first_pub[cid] = d
    sql = f"SELECT {', '.join(_CASE_COLS)} FROM cases"
    if not include_tombstones and has_merge_column(db):
        sql += " WHERE merged_into_id IS NULL"
    out = []
    for row in db.execute(text(sql + " ORDER BY id")):
        c = dict(zip(_CASE_COLS, row))
        idate = c["incident_date"]
        if isinstance(idate, str):
            idate = date.fromisoformat(idate[:10])
        first = c["first_reported_at"]
        if isinstance(first, str):
            first = date.fromisoformat(first[:10])
        out.append(CaseRecord(
            id=c["id"], country=c["country"], victim_name=c["victim_name"], victim_age=c["victim_age"],
            incident_location=c["incident_location"], incident_region=c["incident_region"],
            incident_date=_as_date(idate),
            first_published=first_pub.get(c["id"]) or _as_date(first),
            event_type=c["event_type"], perpetrator_name=c["perpetrator_name"],
            article_count=c["article_count"] or 1, weapon=c["weapon"],
            texts=tuple(titles.get(c["id"], ())) + ((c["summary"],) if c["summary"] else ()),
            article_incident_countries=tuple(inc.get(c["id"], ())),
        ))
    return out


# ── merge / unmerge ──────────────────────────────────────────────────────────

def merge_group(db, group: MergeGroup, batch_id: str) -> List[CaseMergeAudit]:
    """Merge one group. Returns the audit rows (one per absorbed case). Raises
    if the survivor or a member is missing or already merged."""
    survivor = db.get(Case, group.survivor_id)
    if survivor is None or survivor.merged_into_id is not None:
        raise ValueError(f"survivor #{group.survivor_id} missing or already merged")
    before = _snapshot(survivor)
    audits = []
    for mid in group.member_ids:
        if mid == group.survivor_id:
            continue
        absorbed = db.get(Case, mid)
        if absorbed is None or absorbed.merged_into_id is not None:
            raise ValueError(f"member #{mid} missing or already merged")
        moved = [a.id for a in db.query(Article).filter(Article.case_id == mid).order_by(Article.id).all()]
        db.query(Article).filter(Article.case_id == mid).update({Article.case_id: survivor.id}, synchronize_session=False)
        for f in ENRICH_FIELDS:
            if getattr(survivor, f) is None and getattr(absorbed, f) is not None:
                setattr(survivor, f, getattr(absorbed, f))
        if absorbed.summary and (not survivor.summary or len(absorbed.summary) > len(survivor.summary)):
            survivor.summary = absorbed.summary
        if absorbed.first_reported_at and (survivor.first_reported_at is None
                                           or absorbed.first_reported_at < survivor.first_reported_at):
            survivor.first_reported_at = absorbed.first_reported_at
        absorbed.merged_into_id = survivor.id
        audits.append(CaseMergeAudit(
            batch_id=batch_id, tier=group.tier, survivor_id=survivor.id, absorbed_id=mid,
            absorbed_snapshot=json.dumps(_snapshot(absorbed), default=_json_default),
            survivor_before=json.dumps(before, default=_json_default),
            moved_article_ids=json.dumps(moved), reasons=" | ".join(group.reasons),
        ))
    survivor.country = group.country or survivor.country
    survivor.article_count = db.query(Article).filter(Article.case_id == survivor.id).count() or survivor.article_count
    for a in audits:
        db.add(a)
    db.add(survivor)
    db.flush()
    return audits


def apply_groups(db, groups: Iterable[MergeGroup], batch_id: str, tiers=("A",)) -> Dict[str, int]:
    stats = Counter()
    for g in groups:
        if g.tier not in tiers:
            continue
        audits = merge_group(db, g, batch_id)
        stats["groups"] += 1
        stats["absorbed"] += len(audits)
        stats[f"tier_{g.tier}"] += 1
    return dict(stats)


def unmerge_batch(db, batch_id: str) -> int:
    """Undo every non-reverted merge of `batch_id`, newest first. Returns the
    number of absorbed cases restored."""
    rows = db.query(CaseMergeAudit).filter(CaseMergeAudit.batch_id == batch_id,
                                           CaseMergeAudit.reverted_at.is_(None)) \
        .order_by(CaseMergeAudit.id.desc()).all()
    n = 0
    for a in rows:
        absorbed = db.get(Case, a.absorbed_id)
        survivor = db.get(Case, a.survivor_id)
        moved = json.loads(a.moved_article_ids or "[]")
        if moved:
            db.query(Article).filter(Article.id.in_(moved)).update({Article.case_id: a.absorbed_id},
                                                                    synchronize_session=False)
        if absorbed is not None:
            _restore(absorbed, json.loads(a.absorbed_snapshot))
            absorbed.merged_into_id = None
            db.add(absorbed)
        if survivor is not None:
            _restore(survivor, json.loads(a.survivor_before))
            db.add(survivor)
        a.reverted_at = datetime.utcnow()
        db.add(a)
        n += 1
    db.flush()
    return n


def resolve_case_id(db, case_id: int) -> int:
    from .case_linker import resolve_case_id as _r
    return _r(db, case_id, Case)


# ── reporting helpers (shared by the script and its tests) ───────────────────

def distinct_counts(recs: List[CaseRecord], groups: List[MergeGroup], tiers: set,
                    only_ids: Optional[set] = None) -> Counter:
    """(country, year) -> number of distinct cases after merging groups of `tiers`."""
    absorbed, country_of = set(), {}
    for g in groups:
        if g.tier in tiers:
            absorbed.update(m for m in g.member_ids if m != g.survivor_id)
            country_of[g.survivor_id] = g.country
    out = Counter()
    for r in recs:
        if r.id in absorbed or (only_ids is not None and r.id not in only_ids):
            continue
        d = ref_date(r)[0]
        if d is None:
            continue
        c = country_of.get(r.id, r.country)
        out[("UK" if c in ("GB", "UK") else c, d.year)] += 1
    return out
