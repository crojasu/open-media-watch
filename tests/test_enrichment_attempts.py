"""Enrichment records attempts and rotates/backs off
instead of retrying the same failing articles every day. No LLM is called:
`enrich_article` is mocked everywhere."""
from datetime import date, datetime, timedelta
from unittest.mock import patch

from fmedia import enrichment
from fmedia.database import Article


def _art(db, n, **kw):
    a = Article(
        guardian_id=f"enr-{n}", title=f"Incidente {n}", web_url=f"https://x.test/{n}",
        published_date=date(2026, 3, 1), event_type="category_a",
        review_status="confirmed", **kw,
    )
    db.add(a)
    db.commit()
    return a


def _run(db, results, limit=30):
    """Run enrich_pending_articles with enrich_article mocked; returns ids seen."""
    seen = []

    def fake(art, _db):
        seen.append(art.id)
        return results.get(art.id, {"status": "skip", "reason": "scrape_failed"})

    with patch.object(enrichment, "enrich_article", side_effect=fake):
        enrichment.enrich_pending_articles(db, limit=limit, delay=0)
    return seen


def test_records_attempt_and_result_on_failure(db):
    a = _art(db, 1)
    _run(db, {})
    db.refresh(a)
    assert a.enrichment_attempts == 1
    assert a.enrichment_last_result == "scrape_failed"
    assert a.enrichment_last_attempt_at is not None


def test_records_enriched_result(db):
    a = _art(db, 1)
    _run(db, {a.id: {"status": "enriched", "cost": 0.0}})
    db.refresh(a)
    assert a.enrichment_attempts == 1
    assert a.enrichment_last_result == "enriched"


def test_never_tried_come_before_failed_ones(db):
    old_failed = _art(db, 1, enrichment_attempts=1,
                      enrichment_last_attempt_at=datetime.utcnow() - timedelta(days=30))
    never = _art(db, 2)
    # never-tried has lower id on purpose too
    seen = _run(db, {}, limit=1)
    assert seen == [never.id]
    assert old_failed.id not in seen


def test_backoff_skips_recent_failure(db):
    recent = _art(db, 1, enrichment_attempts=1,
                  enrichment_last_attempt_at=datetime.utcnow() - timedelta(hours=1))
    assert _run(db, {}) == []
    assert recent.id


def test_backoff_allows_after_window(db):
    old = _art(db, 1, enrichment_attempts=1,
               enrichment_last_attempt_at=datetime.utcnow() - timedelta(days=10))
    assert _run(db, {}) == [old.id]


def test_gives_up_after_max_attempts(db):
    _art(db, 1, enrichment_attempts=enrichment.MAX_ENRICHMENT_ATTEMPTS,
         enrichment_last_attempt_at=datetime.utcnow() - timedelta(days=365))
    assert _run(db, {}) == []


def test_same_30_not_retried_next_day(db):
    ids = [_art(db, i).id for i in range(1, 41)]
    first = _run(db, {}, limit=30)
    second = _run(db, {}, limit=30)
    assert len(first) == 30
    assert set(first).isdisjoint(second)
    assert set(second) == set(ids) - set(first)


def test_complete_articles_still_excluded(db):
    _art(db, 1, victim_name="A", victim_age=30)
    assert _run(db, {}) == []
