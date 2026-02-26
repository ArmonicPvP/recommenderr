import logging
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from .config import DB_PATH
from .plex import strip_plex_token

log = logging.getLogger(__name__)

_engine: Engine | None = None


def engine() -> Engine:
    global _engine
    if _engine is None:
        log.info("Initializing SQLite engine at %s", DB_PATH)
        _engine = create_engine(f"sqlite:///{DB_PATH}", future=True)
    return _engine


def _column_exists(con, table: str, col: str) -> bool:
    rows = con.exec_driver_sql(f"PRAGMA table_info({table});").fetchall()
    cols = {r[1] for r in rows}
    return col in cols


def _index_exists(con, name: str) -> bool:
    rows = con.exec_driver_sql("PRAGMA index_list(watch_events);").fetchall()
    return any(r[1] == name for r in rows)


def _to_poster_path(value: str | None) -> str | None:
    """Normalize legacy poster_url or thumb values into a token-free path/query string."""
    if not value:
        return value
    cleaned = strip_plex_token(str(value))
    if not cleaned:
        return cleaned
    parts = urlsplit(cleaned)
    if parts.scheme or parts.netloc:
        return urlunsplit(("", "", parts.path, parts.query, parts.fragment))
    return cleaned


def ensure_schema():
    """Create/upgrade schema, idempotent."""
    sql = Path(__file__).with_name("models.sql").read_text(encoding="utf-8")
    log.debug("Ensuring schema")
    with engine().begin() as con:
        for stmt in sql.split(";"):
            s = stmt.strip()
            if s:
                try:
                    con.exec_driver_sql(s + ";")
                except Exception:
                    pass

        if not _column_exists(con, "users", "display_name"):
            log.info("Migrating: add users.display_name")
            con.exec_driver_sql("ALTER TABLE users ADD COLUMN display_name TEXT;")

        try:
            con.exec_driver_sql("UPDATE users SET user_name = NULL WHERE user_name = ''")
        except Exception:
            pass

        item_migrations = [
            ("poster_path", "ALTER TABLE items ADD COLUMN poster_path TEXT"),
            ("content_rating", "ALTER TABLE items ADD COLUMN content_rating TEXT"),
            ("countries_csv", "ALTER TABLE items ADD COLUMN countries_csv TEXT"),
            ("keywords_csv", "ALTER TABLE items ADD COLUMN keywords_csv TEXT"),
            ("tmdb_id", "ALTER TABLE items ADD COLUMN tmdb_id TEXT"),
        ]
        for col, ddl in item_migrations:
            if not _column_exists(con, "items", col):
                log.info("Migrating: add items.%s", col)
                con.exec_driver_sql(ddl)

        has_poster_path = _column_exists(con, "items", "poster_path")
        has_poster_url = _column_exists(con, "items", "poster_url")
        if has_poster_path:
            rows = con.exec_driver_sql(
                """
                SELECT item_id, poster_path, poster_url
                FROM items
                WHERE COALESCE(poster_path, '') != '' OR COALESCE(poster_url, '') != ''
                """
            ).fetchall()
            migrated = 0
            for item_id, poster_path, poster_url in rows:
                source = poster_path if (poster_path and str(poster_path).strip()) else poster_url
                normalized = _to_poster_path(source)
                if has_poster_url:
                    con.exec_driver_sql(
                        "UPDATE items SET poster_path=:pp, poster_url=NULL WHERE item_id=:iid",
                        {"pp": normalized, "iid": str(item_id)},
                    )
                else:
                    con.exec_driver_sql(
                        "UPDATE items SET poster_path=:pp WHERE item_id=:iid",
                        {"pp": normalized, "iid": str(item_id)},
                    )
                migrated += 1
            if migrated:
                log.info("Migrating: normalized %d items poster values into poster_path", migrated)

        if not _column_exists(con, "watch_events", "plex_history_key"):
            log.info("Migrating: add watch_events.plex_history_key")
            con.exec_driver_sql("ALTER TABLE watch_events ADD COLUMN plex_history_key TEXT")

        if not _index_exists(con, "uix_we_source_plex_history_key"):
            log.info("Migrating: create watch_events canonical key unique index")
            con.exec_driver_sql(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS uix_we_source_plex_history_key
                ON watch_events(source, plex_history_key)
                WHERE plex_history_key IS NOT NULL
                """
            )

        if not _index_exists(con, "uix_we_fallback"):
            log.info("Migrating: dedupe legacy watch_events and create fallback unique index")
            deleted = con.exec_driver_sql(
                """
                WITH ranked AS (
                    SELECT id,
                           ROW_NUMBER() OVER (
                             PARTITION BY user_id, item_id, started_at, source
                             ORDER BY id
                           ) AS rn
                    FROM watch_events
                    WHERE plex_history_key IS NULL
                )
                DELETE FROM watch_events
                WHERE id IN (SELECT id FROM ranked WHERE rn > 1)
                """
            ).rowcount
            log.info("Migrating: removed %d duplicate watch_events rows", deleted or 0)
            con.exec_driver_sql(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS uix_we_fallback
                ON watch_events(user_id, item_id, started_at, source)
                WHERE plex_history_key IS NULL
                """
            )
            con.exec_driver_sql("DELETE FROM user_item_pref")
            con.exec_driver_sql(
                """
                INSERT INTO user_item_pref(user_id,item_id,preference,last_seen_at)
                SELECT
                    user_id,
                    item_id,
                    CAST(COUNT(*) AS REAL) AS preference,
                    MAX(started_at) AS last_seen_at
                FROM watch_events
                GROUP BY user_id, item_id
                """
            )
            log.info("Migrating: rebuilt user_item_pref from deduplicated watch_events")

        try:
            con.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_items_year ON items(year)")
            con.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_items_tmdb ON items(tmdb_id)")
            con.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_pref_user ON user_item_pref(user_id)")
        except Exception:
            pass

    log.debug("Schema ensured")
