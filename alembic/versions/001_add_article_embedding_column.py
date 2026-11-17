"""add embedding column to articles

Revision ID: 001
Revises:
Create Date: 2026-07-31

Dialect-conditional:
  - Postgres: adds a real `embedding vector(768)` column, requires the
    `pgvector` extension. We attempt `CREATE EXTENSION IF NOT EXISTS vector`
    inside a SAVEPOINT so a permissions failure there (common on managed DBs
    where the app role isn't a superuser) does NOT abort the outer migration
    transaction — we log a clear warning and keep going. If the extension is
    truly unavailable, the subsequent ADD COLUMN ... vector(768) will fail
    loudly (by design — this is NOT swallowed, so check that your Postgres allows CREATE EXTENSION vector).
  - SQLite (dev): the `embedding_json` TEXT column already exists as a normal
    ORM-mapped column (fmedia/database.py, created via Base.metadata.create_all
    like every other column) — nothing to do here for SQLite, this migration
    is a no-op on that dialect.

init_db()/create_all() is untouched by this migration and keeps working for
local/dev bootstrapping; this migration only manages the Postgres-only vector
column that lives outside the ORM.
"""
from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        # SQLite: embedding_json is a normal ORM column, managed by create_all()
        # and the _migrate_add_typology_columns() self-heal in database.py.
        return

    try:
        with bind.begin_nested():
            op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    except Exception as e:
        print(
            "[alembic 001] WARNING: could not CREATE EXTENSION vector "
            f"(likely missing DB superuser privilege on this Postgres role): {e}\n"
            "[alembic 001] If the extension is not already installed by other means, "
            "the next statement (ADD COLUMN ... vector(768)) will fail — "
            "install the pgvector extension manually (CREATE EXTENSION vector)."
        )

    op.execute("ALTER TABLE articles ADD COLUMN IF NOT EXISTS embedding vector(768)")


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    op.execute("ALTER TABLE articles DROP COLUMN IF EXISTS embedding")
