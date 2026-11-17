"""add victim_age_public and location_public columns to dossiers

Revision ID: 004
Revises: 003
Create Date: 2026-09-15

Bug fixed: Dossier already gated `victim_name` behind `victim_name_public`
(default False, requires an explicit editorial decision to publish), but
`victim_age`, `city`, and `region` were rendered unconditionally on every
published dossier's public page and listing. Age + city + region together
are frequently enough to re-identify a victim even when the name is
deliberately withheld — this closes that gap with the same opt-in pattern.

`victim_age_public` and `location_public` — both `BOOLEAN NOT NULL DEFAULT
false` — gate `Dossier.display_age`/`display_city`/`display_region`
(fmedia/database.py). `city`/`region` themselves stay populated regardless
(used internally for slug generation and case linking); only the two new
flags control what's rendered to the public.

Idempotent by design, same as migrations 002/003: `init_db()` /
Base.metadata.create_all() already adds these columns on every app boot for
a fresh DB, so this migration checks via the inspector before adding them,
rather than assuming a from-scratch DB.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None

TABLE_NAME = "dossiers"
COLUMNS = ["victim_age_public", "location_public"]


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_cols = {c["name"] for c in inspector.get_columns(TABLE_NAME)}
    for col in COLUMNS:
        if col in existing_cols:
            continue
        op.add_column(
            TABLE_NAME,
            sa.Column(col, sa.Boolean(), nullable=False, server_default=sa.false()),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_cols = {c["name"] for c in inspector.get_columns(TABLE_NAME)}
    with op.batch_alter_table(TABLE_NAME) as batch_op:
        for col in COLUMNS:
            if col in existing_cols:
                batch_op.drop_column(col)
