"""create contact_messages table (contact form submissions)

Revision ID: 005
Revises: 004
Create Date: 2026-09-15

Backs the public /contact form: name, email, subject, message, submitted
through POST /api/contact and reviewed by reviewers at GET /admin/contacts.
No email delivery involved — messages are stored and read from the
existing reviewer-gated admin panel, same trust boundary as everything
else there.

Idempotent by design, same as migrations 002-004: init_db() /
Base.metadata.create_all() already creates this table on every app boot
for a fresh DB, so this migration checks via the inspector before
creating it.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "005"
down_revision = "004"
branch_labels = None
depends_on = None

TABLE_NAME = "contact_messages"


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if TABLE_NAME in inspector.get_table_names():
        return
    op.create_table(
        TABLE_NAME,
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("subject", sa.String(length=300), nullable=True),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("read", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_index(
        "ix_contact_messages_created_at", TABLE_NAME, ["created_at"], unique=False
    )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if TABLE_NAME not in inspector.get_table_names():
        return
    op.drop_table(TABLE_NAME)
