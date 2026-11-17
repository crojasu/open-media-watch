"""create case_link_audit table (case-linking agent audit trail, )

Revision ID: 002
Revises: 001
Create Date: 2026-07-31

Fully ORM-mapped table (fmedia/database.py CaseLinkAudit) — unlike migration 001's
Postgres-only `embedding vector(768)` column, this table has the exact same shape
on SQLite and Postgres, so there is no dialect branching here.

Idempotent by design ( F1 required
`alembic upgrade head` to be idempotent; F3 keeps that bar): `init_db()` /
`Base.metadata.create_all()` already creates this table on every app boot (dev
and hosted Postgres alike), so in the common case this migration runs against a DB that
already has the table — we check via SQLAlchemy's inspector before creating it,
rather than assuming a fresh DB the way a from-scratch Alembic project would.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None

TABLE_NAME = "case_link_audit"


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if TABLE_NAME in inspector.get_table_names():
        # Already created by Base.metadata.create_all() (init_db()) — no-op.
        return

    op.create_table(
        TABLE_NAME,
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("article_id", sa.Integer(), sa.ForeignKey("articles.id"), nullable=False),
        sa.Column("decision", sa.String(length=20), nullable=False),
        sa.Column("matched_case_id", sa.Integer(), sa.ForeignKey("cases.id"), nullable=True),
        sa.Column("agent_reasoning", sa.Text(), nullable=True),
        sa.Column("difflib_decision", sa.String(length=20), nullable=True),
        sa.Column("difflib_ratio", sa.Float(), nullable=True),
        sa.Column("iterations", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("tokens_in", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("tokens_out", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("cost_usd", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("model_used", sa.String(length=100), nullable=True),
        sa.Column("mode", sa.String(length=10), nullable=False, server_default="shadow"),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_index(
        "ix_case_link_audit_article_id", TABLE_NAME, ["article_id"], unique=False
    )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if TABLE_NAME not in inspector.get_table_names():
        return
    op.drop_index("ix_case_link_audit_article_id", table_name=TABLE_NAME)
    op.drop_table(TABLE_NAME)
