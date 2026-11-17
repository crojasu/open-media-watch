"""create official_counts table

Revision ID: 006
Revises: 005
Create Date: 2026-10-14

Aggregate yearly relevant incident counts from official/NGO sources per country,
for comparing the volume captured in the database against an external total.
These are bulletin totals, not case-level lists.

Idempotent, same pattern as 002-005: Base.metadata.create_all() already
creates this table on every app boot for a fresh DB; this migration checks
via the inspector before creating it, for environments that only run
`alembic upgrade head`.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "006"
down_revision = "005"
branch_labels = None
depends_on = None

TABLE_NAME = "official_counts"


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if TABLE_NAME in inspector.get_table_names():
        return
    op.create_table(
        TABLE_NAME,
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("country", sa.String(length=2), nullable=False),
        sa.Column("year", sa.Integer(), nullable=False),
        sa.Column("count", sa.Integer(), nullable=False),
        sa.Column("source_name", sa.String(length=200), nullable=True),
        sa.Column("source_url", sa.String(length=500), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
    )
    op.create_index("ix_official_counts_country", TABLE_NAME, ["country"], unique=False)
    op.create_index("ix_official_counts_year", TABLE_NAME, ["year"], unique=False)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if TABLE_NAME not in inspector.get_table_names():
        return
    op.drop_table(TABLE_NAME)
