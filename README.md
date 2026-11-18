# Open Media Watch

An open-source, self-hostable tool for human-rights observatories that want to monitor news coverage with open models. It fetches articles from news sources, filters and classifies them against a taxonomy defined by the observatory, groups articles about the same event into cases, and exposes the result through a REST API, with a sampling audit to track classifier quality over time.

## Status

Early-stage, extracted from Femicide Media Watch, a production system monitoring femicide coverage in six countries. This repository is the starting point; the open-weight classifier, configurable taxonomies, change-detecting archive and deployment package are planned and not yet implemented.

Expect rough edges: the code is a trimmed extraction of a running system, not a polished library.

## What exists today

Everything below is implemented in this repository and covered by the test suite (`pytest`: 217 tests passing on Python 3.12). Real calls to LLM providers, Guardian, GDELT and a PostgreSQL/pgvector database are **not** exercised by the tests or by the quick start.

- **Scheduled ingestion** (`fmedia/jobs.py`): APScheduler jobs for daily fetch, auto-approval of high-confidence classifications, intraday classification runs, a weekly quality audit and an embeddings backfill.
- **Sources** (`fmedia/guardian.py`, `fmedia/countries.py`): a Guardian API client, per-country Google-News-style RSS feeds, and GDELT queries. Search terms and feeds are read from a JSON file (`examples/sources.example.json`, override with `OMW_SOURCES`).
- **Keyword filter and LLM classification cascade** (`fmedia/llm_classifier.py`, `deepseek_classifier.py`, `qwen_classifier.py`, `gemini_triage.py`): Claude is the primary classifier; DeepSeek, Qwen and GPT-4o are tried in turn when the previous provider fails. An optional Gemini triage step can pre-filter. All of these are hosted APIs; **no open-weight/self-hosted model is wired in yet**.
- **Taxonomy and prompts as configuration** (`fmedia/taxonomy.py`): categories, labels and prompts are loaded from a JSON file plus prompt text files (`examples/taxonomy.example.json`, override with `OMW_TAXONOMY`). The bundled example is a deliberately generic placeholder (`category_a` … `category_d`, `not_relevant`) with a minimal example prompt. It is not validated against a schema yet and gives only basic classification quality.
- **Article-to-case linking** (`fmedia/cases.py`, `case_linker.py`): `find_matching_case()` and a second-generation rule-based linker (victim/perpetrator names, location, dates, title similarity, country guards), plus a reversible merge pass with an audit trail (`case_merge.py`, `case_merge_apply.py`). Embeddings (`fmedia/embeddings.py`, Gemini) are generated for confirmed articles by a nightly job; the linking rules themselves do not use embedding similarity.
- **Sampling quality audit and drift calculation** (`fmedia/quality_audit.py`): re-classifies a random sample of automatically classified articles with Claude and reports the discrepancy rate; a rate above `DRIFT_THRESHOLD` (0.15) raises an alert. It is not a human-review workflow.
- **Data models** (`fmedia/database.py`): `Article`, `Case`, `Dossier`, `DossierEntry`, `DossierFact` and audit tables, with Alembic migrations (`alembic/`).
- **Public REST API** (`fmedia/main.py`, `fmedia/dossiers.py`): `/health`, `/api/articles` (public fields only), `/api/timeline`, `/api/types`, `/api/relaciones`, `/api/motivos`, `/api/types-by-month`, `/api/sankey`, `GET /api/dossiers` and `GET /api/dossiers/by-slug/{slug}`. Write endpoints for dossiers require HTTP Basic credentials (`REVIEW_USERS`). Interactive docs at `/docs`.
- **Packaging**: `requirements.txt`, a `Dockerfile` (not built or tested yet), and `scripts/seed_dossiers.py`, which seeds two fictitious dossiers.

### Naming note

Schema and identifiers use neutral names (`event_type`, `event_relation`, `event_motive`, `event_summary`, `uses_topic_term`). Some API paths (`/api/relaciones`, `/api/motivos`) and some source-language strings and enum values (Spanish, e.g. dossier statuses) are inherited from the original system and have not been generalised yet; that belongs to the planned taxonomy work.

## Roadmap (all planned, none implemented)

1. **Configurable taxonomy**: a validated schema for categories, labels and prompts, and neutral schema names.
2. **Self-hosted open-model classification and benchmark**: run open-weight models locally and measure them against the hosted cascade on labelled data.
3. **Full-text archive with change detection**: store article text with content hashes and detect later edits.
4. **Deployment package and documentation**: a reproducible, documented way to self-host.

## Quick start

Tested on Python 3.12 in a clean virtual environment.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

cp .env.example .env
# edit .env: set REVIEW_USERS to a long random value; ANTHROPIC_API_KEY is
# required by startup validation (a placeholder is enough to start the API,
# a real key is needed to actually classify)

python run.py --prod --port 8765
```

In another terminal:

```bash
curl http://localhost:8765/health
curl http://localhost:8765/api/articles
```

Run the tests and (optionally) seed two fictitious dossiers:

```bash
pytest
python scripts/seed_dossiers.py
```

By default the app uses a local SQLite file at `data/omw.db`. Set `DATABASE_URL` to a PostgreSQL URL for anything beyond a trial; embedding-based linking needs PostgreSQL with pgvector, which is untested here.

## Ethics and data protection

Monitoring coverage of violence touches real people, so some constraints are enforced in the schema rather than left to convention. To be precise about what they cover:

- `Dossier`, `DossierEntry` and `DossierFact` (the public-facing memorial/analysis layer) have **no perpetrator-name field** and **no fields that identify minors**. A database `CheckConstraint` (`ck_entry_quote_requires_attribution`) rejects quoted entries without attribution.
- On dossiers, the victim's name, age and location are only exposed when explicit `*_public` flags are set, and images only when `image_rights_verified` is true. Facts carry a status (`hecho_judicial`, `alegacion_fiscalia`, `version_medio_sin_confirmar`) so allegations are not presented as established.
- The internal `Article` and `Case` tables **do** have extraction fields such as `victim_name`, `perpetrator_name`, `perpetrator_age`, `perpetrator_status` and `children_orphaned`. They are not returned by the public article endpoint, but anyone deploying this tool is responsible for the legal basis, access control and retention of that data in their own jurisdiction. Which personal fields are extracted is meant to become configurable as part of the taxonomy work.

Femicide Media Watch's curated corpus (articles, cases, names, labels), its tuned prompts and keyword lists are **not** part of this repository. The repository ships only generic example configuration and fictitious test data.

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md).

## License

GNU Affero General Public License v3.0, see [LICENSE](LICENSE). Copyright (C) 2026 Catalina Rojas Ugarte (see [NOTICE](NOTICE)).
