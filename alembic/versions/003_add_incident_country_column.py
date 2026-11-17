"""add incident_country column to articles

Revision ID: 003
Revises: 002
Create Date: 2026-07-31

Bug fixed: a media outlet monitored under
`articles.country` (e.g. "UK") can report a relevant incident that actually happened in
a DIFFERENT country (e.g. India) — the classifier previously had no field to
record where the incident occurred, so ≥527 UK-confirmed articles turned out
to describe relevant incidents abroad, all counted as "UK" cases.

`incident_country VARCHAR(2)` nullable — ISO-2 of the country where the
incident ACTUALLY happened, as determined by the LLM classifier. NULL means
"not determined" or "same as `country`" (both read as local by
`fmedia.database.local_incident_filter()`); a different ISO-2 code marks the
article as a genuine foreign incident.

Dialect-agnostic, plain column — unlike migration 001's Postgres-only
`embedding vector(768)` column, this one is ALSO a normal ORM-mapped column on
Article (fmedia/database.py), created by Base.metadata.create_all() on
SQLite/dev like every other typed column, and self-healed there too via
_migrate_add_typology_columns() for already-running Postgres deployments that
boot before `alembic upgrade head` runs. This migration exists so a fresh
`alembic upgrade head` (no app boot in between) also gets the column — same
idempotent-by-inspection pattern as migration 002.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None

TABLE_NAME = "articles"
COLUMN_NAME = "incident_country"


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_cols = {c["name"] for c in inspector.get_columns(TABLE_NAME)}
    if COLUMN_NAME in existing_cols:
        # Already created by Base.metadata.create_all() (init_db()) or the
        # ad-hoc self-heal migrator — no-op.
        return
    op.add_column(TABLE_NAME, sa.Column(COLUMN_NAME, sa.String(length=2), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_cols = {c["name"] for c in inspector.get_columns(TABLE_NAME)}
    if COLUMN_NAME not in existing_cols:
        return
    with op.batch_alter_table(TABLE_NAME) as batch_op:
        batch_op.drop_column(COLUMN_NAME)
