"""add cases.merged_into_id tombstone column

Revision ID: 011
Revises: 010
Create Date: 2026-10-16

NULL for every live case. Set to the survivor's id when a case is merged by
scripts/case_merge_apply.py; listings and the coverage numerator skip rows
where it is not NULL and the linker follows the chain to the survivor.

Idempotent: checks the inspector before ALTER so a DB bootstrapped by
database._migrate_add_case_merge_column() is left alone. Only touches the
column when missing (ADD COLUMN IF NOT EXISTS still takes an ACCESS EXCLUSIVE
lock on Postgres).
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "011"
down_revision = "010"
branch_labels = None
depends_on = None

TABLE_NAME = "cases"
COLUMN = "merged_into_id"
INDEX = "ix_cases_merged_into_id"


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    cols = {c["name"] for c in inspector.get_columns(TABLE_NAME)}
    if COLUMN not in cols:
        op.add_column(TABLE_NAME, sa.Column(COLUMN, sa.Integer(), sa.ForeignKey("cases.id"), nullable=True))
    idx = {i["name"] for i in sa.inspect(bind).get_indexes(TABLE_NAME)}
    if INDEX not in idx:
        op.create_index(INDEX, TABLE_NAME, [COLUMN], unique=False)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    idx = {i["name"] for i in inspector.get_indexes(TABLE_NAME)}
    if INDEX in idx:
        op.drop_index(INDEX, table_name=TABLE_NAME)
    cols = {c["name"] for c in inspector.get_columns(TABLE_NAME)}
    if COLUMN in cols:
        op.drop_column(TABLE_NAME, COLUMN)
