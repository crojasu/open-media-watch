"""Startup migrations must not take a table lock when there is nothing to migrate.

`ALTER TABLE ... ADD COLUMN IF NOT EXISTS` needs an ACCESS EXCLUSIVE lock even when the column
already exists, so a single open transaction on `articles` made every deploy hang before the
server started listening (production outage, 2026-10-16)."""
from sqlalchemy import event

from fmedia import database


def _count_ddl(fn) -> list[str]:
    seen: list[str] = []

    def spy(conn, cursor, statement, params, context, executemany):
        if statement.lstrip().upper().startswith(("ALTER", "CREATE INDEX", "SET ")):
            seen.append(statement.strip().split("\n")[0][:80])

    event.listen(database.engine, "before_cursor_execute", spy)
    try:
        fn()
    finally:
        event.remove(database.engine, "before_cursor_execute", spy)
    return seen


def test_typology_migration_issues_no_ddl_when_columns_exist():
    assert _count_ddl(database._migrate_add_typology_columns) == []


def test_fingerprint_migration_issues_no_ddl_when_column_and_index_exist():
    database._migrate_add_fingerprint_column()  # first run may create the index on a bare test DB
    assert _count_ddl(database._migrate_add_fingerprint_column) == []


def test_missing_column_is_still_added():
    from sqlalchemy import text

    with database.engine.begin() as conn:
        conn.execute(text("ALTER TABLE articles DROP COLUMN weapon"))
    assert "weapon" not in {c["name"] for c in __import__("sqlalchemy").inspect(database.engine).get_columns("articles")}
    database._migrate_add_typology_columns()
    assert "weapon" in {c["name"] for c in __import__("sqlalchemy").inspect(database.engine).get_columns("articles")}
