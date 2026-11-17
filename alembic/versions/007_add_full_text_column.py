"""add full_text column to articles (persist scraped article bodies)

Revision ID: 007
Revises: 006
Create Date: 2026-10-15

fmedia/enrichment.py's scrape_article() already downloads the full article
body to re-extract victim data with Claude/DeepSeek — but only ever used it
transiently inside that one call, then threw it away. Only `content_snippet`
(really just the RSS title+summary, ~80-100 chars) was ever persisted, which
starved case-linking (victim_name/incident_date/incident_location extraction)
and any future retraining on richer text than a headline.

`full_text TEXT` nullable — populated going forward whenever enrichment scrapes
an article; backfilling it for already-confirmed articles is a separate,
later pass (re-scraping ~24k articles, not part of this migration).

Same idempotent-by-inspection pattern as migration 003.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "007"
down_revision = "006"
branch_labels = None
depends_on = None

TABLE_NAME = "articles"
COLUMN_NAME = "full_text"


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_cols = {c["name"] for c in inspector.get_columns(TABLE_NAME)}
    if COLUMN_NAME in existing_cols:
        return
    op.add_column(TABLE_NAME, sa.Column(COLUMN_NAME, sa.Text(), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_cols = {c["name"] for c in inspector.get_columns(TABLE_NAME)}
    if COLUMN_NAME not in existing_cols:
        return
    with op.batch_alter_table(TABLE_NAME) as batch_op:
        batch_op.drop_column(COLUMN_NAME)
