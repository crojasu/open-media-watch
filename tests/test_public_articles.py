"""GET /api/articles is public: it must never expose reviewer/label columns."""
from datetime import date

from fmedia.database import Article

PUBLIC_KEYS = {
    "id", "title", "web_url", "section", "published_date", "source",
    "country", "snippet", "event_type", "event_relation", "event_motive",
}
INTERNAL_KEYS = {"review_status", "reviewed_by", "reviewed_at", "human_label"}


def test_public_articles_hide_reviewer_fields(client, db):
    db.add(Article(
        guardian_id="gid-pub-1", title="Titular de prueba", published_date=date.today(),
        country="MX", source="Fuente Prueba", web_url="https://example.test/1",
        content_snippet="Snippet de prueba", review_status="confirmed",
        reviewed_by="alguien", human_label=1, event_type="category_a",
    ))
    db.commit()

    items = client.get("/api/articles").json()["items"]

    assert len(items) == 1
    assert set(items[0]) == PUBLIC_KEYS
    assert INTERNAL_KEYS.isdisjoint(items[0])
