"""create stage_daily_counts table (per-stage pipeline observability)

Revision ID: 009
Revises: 008
Create Date: 2026-10-16

One row per (day, country) with the daily counts of each pipeline stage
(fetched / kept / confirmed / enriched / with_victim_name / embedded), derived
from existing articles columns by fmedia/stage_metrics.py. Used for the
day-over-day drop alert and GET /admin/stage-counts.

Idempotent, same pattern as 002-006: Base.metadata.create_all() already
creates this table on app boot; this migration checks via the inspector first.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "009"
down_revision = "008"
branch_labels = None
depends_on = None

TABLE_NAME = "stage_daily_counts"


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if TABLE_NAME in inspector.get_table_names():
        return
    op.create_table(
        TABLE_NAME,
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column("country", sa.String(length=2), nullable=False),
        sa.Column("fetched", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("kept", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("confirmed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("enriched", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("with_victim_name", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("embedded", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("computed_at", sa.DateTime(), nullable=True),
        sa.UniqueConstraint("day", "country", name="uq_stage_daily_day_country"),
    )
    op.create_index("ix_stage_daily_counts_day", TABLE_NAME, ["day"], unique=False)
    op.create_index("ix_stage_daily_counts_country", TABLE_NAME, ["country"], unique=False)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if TABLE_NAME not in inspector.get_table_names():
        return
    op.drop_table(TABLE_NAME)
