"""Unit tests for fmedia/embeddings.py ."""
import json
from datetime import date

import pytest

from fmedia import embeddings as emb
from fmedia.database import Article


def _make_article(db, **overrides):
    defaults = dict(
        guardian_id=overrides.pop("guardian_id", "emb-test"),
        title="Test article",
        published_date=date(2026, 1, 1),
        country="CL",
        review_status="confirmed",
    )
    defaults.update(overrides)
    art = Article(**defaults)
    db.add(art)
    db.commit()
    db.refresh(art)
    return art


# ── build_embedding_text — truncation + fallback chain ────────────────────────

def test_build_embedding_text_combines_title_and_summary():
    text = emb.build_embedding_text("Title", event_summary="Summary here", content_snippet="Snippet")
    assert text == "Title Summary here"


def test_build_embedding_text_falls_back_to_snippet_without_summary():
    text = emb.build_embedding_text("Title", event_summary=None, content_snippet="Snippet")
    assert text == "Title Snippet"


def test_build_embedding_text_prefers_summary_over_snippet():
    text = emb.build_embedding_text("Title", event_summary="Summary", content_snippet="Snippet")
    assert "Summary" in text and "Snippet" not in text


def test_build_embedding_text_truncates_to_max_chars():
    long_summary = "x" * 5000
    text = emb.build_embedding_text("Title", event_summary=long_summary)
    assert len(text) == emb.MAX_TEXT_CHARS


def test_build_embedding_text_handles_missing_title():
    text = emb.build_embedding_text(None, event_summary="Only summary")
    assert text == "Only summary"


def test_build_embedding_text_handles_all_none():
    assert emb.build_embedding_text(None, None, None) == ""


# ── cosine_similarity ──────────────────────────────────────────────────────────

def test_cosine_similarity_identical_vectors_is_one():
    v = [1.0, 2.0, 3.0]
    assert emb.cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero():
    assert emb.cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors_is_negative_one():
    assert emb.cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector_returns_zero():
    assert emb.cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_cosine_similarity_mismatched_length_returns_zero():
    assert emb.cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0


def test_cosine_similarity_empty_vectors_returns_zero():
    assert emb.cosine_similarity([], []) == 0.0


# ── embed_text cascade: Gemini primary, OpenAI fallback (mocks) ───────────────

def test_embed_text_returns_none_without_any_api_key(monkeypatch):
    monkeypatch.setattr(emb, "GEMINI_API_KEY", "")
    monkeypatch.setattr(emb, "OPENAI_API_KEY", "")
    assert emb.embed_text("hello") is None


def test_embed_text_returns_none_for_blank_input():
    assert emb.embed_text("") is None
    assert emb.embed_text("   ") is None


def test_embed_text_uses_gemini_when_available(monkeypatch):
    monkeypatch.setattr(emb, "GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(emb, "_embed_gemini", lambda t: [0.1, 0.2, 0.3])
    calls = []
    monkeypatch.setattr(emb, "_embed_openai", lambda t: calls.append(t) or [9.0])
    result = emb.embed_text("hello")
    assert result == [0.1, 0.2, 0.3]
    assert calls == []  # fallback never invoked when Gemini succeeds


def test_embed_text_falls_back_to_openai_when_gemini_fails(monkeypatch):
    monkeypatch.setattr(emb, "GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(emb, "OPENAI_API_KEY", "fake-key")
    monkeypatch.setattr(emb, "_embed_gemini", lambda t: None)
    monkeypatch.setattr(emb, "_embed_openai", lambda t: [0.4, 0.5])
    result = emb.embed_text("hello")
    assert result == [0.4, 0.5]


def test_embed_text_returns_none_when_both_providers_fail(monkeypatch):
    monkeypatch.setattr(emb, "GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(emb, "OPENAI_API_KEY", "fake-key")
    monkeypatch.setattr(emb, "_embed_gemini", lambda t: None)
    monkeypatch.setattr(emb, "_embed_openai", lambda t: None)
    assert emb.embed_text("hello") is None


def test_embed_gemini_returns_none_without_api_key(monkeypatch):
    monkeypatch.setattr(emb, "GEMINI_API_KEY", "")
    assert emb._embed_gemini("hello") is None


def test_embed_gemini_parses_sdk_response(monkeypatch):
    monkeypatch.setattr(emb, "GEMINI_API_KEY", "fake-key")

    class FakeEmbedding:
        values = [0.1, 0.2, 0.3]

    class FakeResponse:
        embeddings = [FakeEmbedding()]

    class FakeModels:
        def embed_content(self, model, contents, config):
            assert model == emb.GEMINI_EMBED_MODEL
            assert config.output_dimensionality == emb.EMBED_DIM
            return FakeResponse()

    class FakeClient:
        def __init__(self, api_key):
            self.models = FakeModels()

    import google.genai as genai_module
    monkeypatch.setattr(genai_module, "Client", FakeClient)

    assert emb._embed_gemini("some text") == [0.1, 0.2, 0.3]


def test_embed_gemini_returns_none_on_sdk_error(monkeypatch):
    monkeypatch.setattr(emb, "GEMINI_API_KEY", "fake-key")

    class FailingClient:
        def __init__(self, api_key):
            raise RuntimeError("boom")

    import google.genai as genai_module
    monkeypatch.setattr(genai_module, "Client", FailingClient)
    assert emb._embed_gemini("text") is None


def test_embed_openai_returns_none_without_api_key(monkeypatch):
    monkeypatch.setattr(emb, "OPENAI_API_KEY", "")
    assert emb._embed_openai("hello") is None


def test_embed_openai_parses_sdk_response(monkeypatch):
    monkeypatch.setattr(emb, "OPENAI_API_KEY", "fake-key")

    class FakeEmbeddingItem:
        embedding = [0.7, 0.8]

    class FakeEmbeddingsResp:
        data = [FakeEmbeddingItem()]

    class FakeEmbeddingsAPI:
        def create(self, model, input, dimensions):
            assert model == emb.OPENAI_EMBED_MODEL
            assert dimensions == emb.EMBED_DIM
            return FakeEmbeddingsResp()

    class FakeOpenAIClient:
        def __init__(self, api_key):
            self.embeddings = FakeEmbeddingsAPI()

    import openai as openai_module
    monkeypatch.setattr(openai_module, "OpenAI", FakeOpenAIClient)

    assert emb._embed_openai("some text") == [0.7, 0.8]


def test_embed_openai_returns_none_on_sdk_error(monkeypatch):
    monkeypatch.setattr(emb, "OPENAI_API_KEY", "fake-key")

    class FailingClient:
        def __init__(self, api_key):
            raise RuntimeError("boom")

    import openai as openai_module
    monkeypatch.setattr(openai_module, "OpenAI", FailingClient)
    assert emb._embed_openai("text") is None


# ── BruteForceStore (SQLite backend) ───────────────────────────────────────────

def test_brute_force_store_ranks_by_cosine_similarity(db):
    a = _make_article(db, guardian_id="bf-a", title="Close match")
    b = _make_article(db, guardian_id="bf-b", title="Far match")
    store = emb.BruteForceStore(db)
    store.save_embedding(a, [1.0, 0.0, 0.0])
    store.save_embedding(b, [0.0, 1.0, 0.0])
    db.commit()

    results = store.search([1.0, 0.0, 0.0], k=10)
    assert [r["article"].id for r in results] == [a.id, b.id]
    assert results[0]["score"] == pytest.approx(1.0)
    assert results[1]["score"] == pytest.approx(0.0)


def test_brute_force_store_excludes_non_confirmed(db):
    a = _make_article(db, guardian_id="bf-c", review_status="confirmed")
    b = _make_article(db, guardian_id="bf-d", review_status="pending")
    store = emb.BruteForceStore(db)
    store.save_embedding(a, [1.0, 0.0])
    store.save_embedding(b, [1.0, 0.0])
    db.commit()

    results = store.search([1.0, 0.0], k=10)
    assert [r["article"].id for r in results] == [a.id]


def test_brute_force_store_excludes_articles_without_embedding(db):
    a = _make_article(db, guardian_id="bf-e")
    emb.BruteForceStore(db).save_embedding(a, [1.0, 0.0])
    _make_article(db, guardian_id="bf-f")  # no embedding saved
    db.commit()

    results = emb.BruteForceStore(db).search([1.0, 0.0], k=10)
    assert len(results) == 1
    assert results[0]["article"].id == a.id


def test_brute_force_store_filters_by_country(db):
    a = _make_article(db, guardian_id="bf-g", country="CL")
    b = _make_article(db, guardian_id="bf-h", country="AR")
    store = emb.BruteForceStore(db)
    store.save_embedding(a, [1.0, 0.0])
    store.save_embedding(b, [1.0, 0.0])
    db.commit()

    results = store.search([1.0, 0.0], country="CL")
    assert [r["article"].id for r in results] == [a.id]


def test_brute_force_store_filters_by_year(db):
    a = _make_article(db, guardian_id="bf-i", published_date=date(2025, 5, 1))
    b = _make_article(db, guardian_id="bf-j", published_date=date(2026, 5, 1))
    store = emb.BruteForceStore(db)
    store.save_embedding(a, [1.0, 0.0])
    store.save_embedding(b, [1.0, 0.0])
    db.commit()

    results = store.search([1.0, 0.0], year=2026)
    assert [r["article"].id for r in results] == [b.id]


def test_brute_force_store_respects_k_limit(db):
    store = emb.BruteForceStore(db)
    for i in range(5):
        a = _make_article(db, guardian_id=f"bf-k-{i}")
        store.save_embedding(a, [1.0, 0.0])
    db.commit()

    results = store.search([1.0, 0.0], k=2)
    assert len(results) == 2


def test_get_vector_store_picks_bruteforce_for_sqlite(db):
    assert isinstance(emb.get_vector_store(db), emb.BruteForceStore)


# ── vectors_for_status — raw-vector reads ───────────

def test_vectors_for_status_returns_confirmed_with_embedding(db):
    a = _make_article(db, guardian_id="vfs-a", review_status="confirmed")
    _make_article(db, guardian_id="vfs-b", review_status="pending")
    store = emb.BruteForceStore(db)
    store.save_embedding(a, [1.0, 0.0])
    db.commit()

    rows = store.vectors_for_status("confirmed")
    assert [r["article"].id for r in rows] == [a.id]
    assert rows[0]["vector"] == [1.0, 0.0]


def test_vectors_for_status_filters_by_review_status(db):
    confirmed = _make_article(db, guardian_id="vfs-c", review_status="confirmed")
    pending = _make_article(db, guardian_id="vfs-d", review_status="pending")
    store = emb.BruteForceStore(db)
    store.save_embedding(confirmed, [1.0, 0.0])
    store.save_embedding(pending, [0.0, 1.0])
    db.commit()

    rows = store.vectors_for_status("pending")
    assert [r["article"].id for r in rows] == [pending.id]


def test_vectors_for_status_filters_by_country(db):
    a = _make_article(db, guardian_id="vfs-e", country="CL")
    b = _make_article(db, guardian_id="vfs-f", country="AR")
    store = emb.BruteForceStore(db)
    store.save_embedding(a, [1.0, 0.0])
    store.save_embedding(b, [0.0, 1.0])
    db.commit()

    rows = store.vectors_for_status("confirmed", country="CL")
    assert [r["article"].id for r in rows] == [a.id]


def test_vectors_for_status_excludes_without_embedding(db):
    a = _make_article(db, guardian_id="vfs-g")
    emb.BruteForceStore(db).save_embedding(a, [1.0, 0.0])
    _make_article(db, guardian_id="vfs-h")  # no embedding saved
    db.commit()

    rows = emb.BruteForceStore(db).vectors_for_status("confirmed")
    assert len(rows) == 1
    assert rows[0]["article"].id == a.id


def test_vectors_for_status_respects_limit(db):
    store = emb.BruteForceStore(db)
    for i in range(4):
        a = _make_article(db, guardian_id=f"vfs-limit-{i}")
        store.save_embedding(a, [1.0, 0.0])
    db.commit()

    rows = store.vectors_for_status("confirmed", limit=2)
    assert len(rows) == 2


def test_vectors_for_status_local_only_excludes_foreign_incidents(db):
    local = _make_article(db, guardian_id="vfs-local", country="CL", incident_country=None)
    foreign = _make_article(db, guardian_id="vfs-foreign", country="CL", incident_country="AR")
    store = emb.BruteForceStore(db)
    store.save_embedding(local, [1.0, 0.0])
    store.save_embedding(foreign, [0.0, 1.0])
    db.commit()

    rows = store.vectors_for_status("confirmed", country="CL", local_only=True)
    assert [r["article"].id for r in rows] == [local.id]

    rows_all = store.vectors_for_status("confirmed", country="CL", local_only=False)
    assert {r["article"].id for r in rows_all} == {local.id, foreign.id}


# ── _parse_vector_literal — inverse of _vector_literal (PgVectorStore parsing) ─

def test_parse_vector_literal_round_trips_vector_literal():
    original = [0.1, -0.2, 3.0]
    literal = emb._vector_literal(original)
    assert emb._parse_vector_literal(literal) == pytest.approx(original)


def test_parse_vector_literal_returns_none_for_empty_or_none():
    assert emb._parse_vector_literal(None) is None
    assert emb._parse_vector_literal("") is None


def test_parse_vector_literal_returns_empty_list_for_empty_brackets():
    assert emb._parse_vector_literal("[]") == []


def test_parse_vector_literal_returns_none_for_malformed_input():
    assert emb._parse_vector_literal("[abc,def]") is None


# ── embed_article — the confirm-article hook entry point ─────────────────────

def test_embed_article_persists_embedding_and_commits(db, monkeypatch):
    a = _make_article(db, guardian_id="ea-1")
    monkeypatch.setattr(emb, "embed_text", lambda t: [0.1, 0.2])
    assert emb.embed_article(db, a) is True
    db.refresh(a)
    assert json.loads(a.embedding_json) == [0.1, 0.2]


def test_embed_article_returns_false_when_no_embedding_available(db, monkeypatch):
    a = _make_article(db, guardian_id="ea-2")
    monkeypatch.setattr(emb, "embed_text", lambda t: None)
    assert emb.embed_article(db, a) is False
    db.refresh(a)
    assert a.embedding_json is None


def test_embed_article_never_raises_on_internal_error(db, monkeypatch):
    a = _make_article(db, guardian_id="ea-3")

    def boom(t):
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(emb, "embed_text", boom)
    assert emb.embed_article(db, a) is False  # must not raise


# ── get_embedding_counts / run_embeddings_backfill (admin manual job) ────────

def test_get_embedding_counts_reports_missing_and_embedded(db):
    a = _make_article(db, guardian_id="cnt-a")
    _make_article(db, guardian_id="cnt-b")
    emb.BruteForceStore(db).save_embedding(a, [1.0, 0.0])
    db.commit()

    counts = emb.get_embedding_counts(db)
    assert counts == {"confirmed_total": 2, "missing": 1, "embedded": 1}


def test_run_embeddings_backfill_embeds_missing_confirmed_articles(db, monkeypatch):
    monkeypatch.setattr(emb, "embed_text", lambda text: [1.0, 0.0])
    a = _make_article(db, guardian_id="rb-a")
    _make_article(db, guardian_id="rb-b", review_status="pending")  # not confirmed — excluded

    stats = emb.run_embeddings_backfill()
    assert stats == {"total": 1, "embedded": 1, "failed": 0}
    db.refresh(a)
    assert a.embedding_json is not None


def test_run_embeddings_backfill_never_reprocesses_already_embedded(db, monkeypatch):
    monkeypatch.setattr(emb, "embed_text", lambda text: [1.0, 0.0])
    _make_article(db, guardian_id="rb-c")
    first_stats = emb.run_embeddings_backfill()
    assert first_stats["embedded"] == 1

    def must_not_be_called(text):
        raise AssertionError("already-embedded article should not be re-queried")

    monkeypatch.setattr(emb, "embed_text", must_not_be_called)
    second_stats = emb.run_embeddings_backfill()
    assert second_stats == {"total": 0, "embedded": 0, "failed": 0}


def test_run_embeddings_backfill_respects_limit_oldest_id_first(db, monkeypatch):
    monkeypatch.setattr(emb, "embed_text", lambda text: [1.0, 0.0])
    a = _make_article(db, guardian_id="rb-d")
    b = _make_article(db, guardian_id="rb-e")

    stats = emb.run_embeddings_backfill(limit=1)
    assert stats["total"] == 1
    db.refresh(a)
    db.refresh(b)
    assert a.embedding_json is not None  # lower id processed first
    assert b.embedding_json is None


def test_run_embeddings_backfill_resets_running_flag_when_done(db, monkeypatch):
    monkeypatch.setattr(emb, "embed_text", lambda text: [1.0, 0.0])
    _make_article(db, guardian_id="rb-f")
    emb.run_embeddings_backfill()
    assert emb.is_backfill_running() is False
    assert emb.last_backfill_result() == {"total": 1, "embedded": 1, "failed": 0}
