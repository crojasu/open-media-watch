"""create case_merge_audit table (reversible case merges)

Revision ID: 010
Revises: 009
Create Date: 2026-10-16

One row per ABSORBED case of a merge: survivor, snapshot of the absorbed case,
the survivor fields before the merge, and the article ids that moved, so a
batch can be undone exactly (fmedia/case_merge_apply.py::unmerge_batch).

Idempotent, same pattern as 002-009: Base.metadata.create_all() already creates
this table on app boot; this migration checks via the inspector first.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "010"
down_revision = "009"
branch_labels = None
depends_on = None

TABLE_NAME = "case_merge_audit"


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if TABLE_NAME in inspector.get_table_names():
        return
    op.create_table(
        TABLE_NAME,
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("batch_id", sa.String(length=64), nullable=False),
        sa.Column("tier", sa.String(length=1), nullable=False),
        sa.Column("survivor_id", sa.Integer(), sa.ForeignKey("cases.id"), nullable=False),
        sa.Column("absorbed_id", sa.Integer(), sa.ForeignKey("cases.id"), nullable=False),
        sa.Column("absorbed_snapshot", sa.Text(), nullable=False),
        sa.Column("survivor_before", sa.Text(), nullable=False),
        sa.Column("moved_article_ids", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("reasons", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("reverted_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_case_merge_audit_batch_id", TABLE_NAME, ["batch_id"], unique=False)
    op.create_index("ix_case_merge_audit_survivor_id", TABLE_NAME, ["survivor_id"], unique=False)
    op.create_index("ix_case_merge_audit_absorbed_id", TABLE_NAME, ["absorbed_id"], unique=False)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if TABLE_NAME not in inspector.get_table_names():
        return
    op.drop_table(TABLE_NAME)
