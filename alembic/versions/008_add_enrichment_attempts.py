"""add enrichment attempt tracking columns to articles

Revision ID: 008
Revises: 007
Create Date: 2026-10-16

enrich_pending_articles used to run limit=30/day ordered by id desc without
recording anything, so it retried the same failing articles forever. These
columns let it rotate (never-tried first) and back off / give up per article.

Same idempotent-by-inspection pattern as migrations 003/007.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "008"
down_revision = "007"
branch_labels = None
depends_on = None

TABLE_NAME = "articles"
COLUMNS = [
    ("enrichment_attempts", lambda: sa.Column("enrichment_attempts", sa.Integer(), nullable=True, server_default="0")),
    ("enrichment_last_attempt_at", lambda: sa.Column("enrichment_last_attempt_at", sa.DateTime(), nullable=True)),
    ("enrichment_last_result", lambda: sa.Column("enrichment_last_result", sa.String(40), nullable=True)),
]


def upgrade() -> None:
    existing = {c["name"] for c in sa.inspect(op.get_bind()).get_columns(TABLE_NAME)}
    for name, make in COLUMNS:
        if name not in existing:
            op.add_column(TABLE_NAME, make())


def downgrade() -> None:
    existing = {c["name"] for c in sa.inspect(op.get_bind()).get_columns(TABLE_NAME)}
    with op.batch_alter_table(TABLE_NAME) as batch_op:
        for name, _ in reversed(COLUMNS):
            if name in existing:
                batch_op.drop_column(name)
