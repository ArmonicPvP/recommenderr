import logging
import math
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from .config import DB_PATH

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


def _migration_done(con, name: str) -> bool:
    row = con.exec_driver_sql(
        "SELECT 1 FROM schema_migrations WHERE name=:name LIMIT 1", {"name": name}
    ).fetchone()
    return row is not None


def _mark_migration_done(con, name: str):
    con.exec_driver_sql(
        "INSERT OR IGNORE INTO schema_migrations(name) VALUES (:name)", {"name": name}
    )


def _preference(grp: pd.DataFrame) -> float:
    events = len(grp)
    started = pd.to_datetime(grp["started_at"], utc=True, errors="coerce")
    last_seen = started.max()
    days = 999.0 if pd.isna(last_seen) else (pd.Timestamp.utcnow() - last_seen).days
    recent = math.exp(-days / 90.0)
    dur = grp["duration"].dropna()
    dur_score = min(dur.max() / 1800.0, 1.0) if not dur.empty else 0.5
    w_amt, w_rec, w_dur, w_bias = 0.55, 0.30, 0.10, 0.05
    return w_amt * math.log1p(events) + w_rec * recent + w_dur * dur_score + w_bias


def _recompute_user_item_pref(con):
    df_we = pd.read_sql_query(
        "SELECT user_id,item_id,started_at,duration FROM watch_events", con
    )
    con.exec_driver_sql("DELETE FROM user_item_pref")
    if df_we.empty:
        return 0

    rows = []
    for (user_id, item_id), grp in df_we.groupby(["user_id", "item_id"]):
        rows.append(
            {
                "user_id": user_id,
                "item_id": item_id,
                "preference": _preference(grp),
                "last_seen_at": pd.to_datetime(
                    grp["started_at"], utc=True, errors="coerce"
                ).max(),
            }
        )

    df_pref = pd.DataFrame(rows)
    df_pref["last_seen_at"] = df_pref["last_seen_at"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    for _, r in df_pref.iterrows():
        con.exec_driver_sql(
            """
            INSERT INTO user_item_pref(user_id,item_id,preference,last_seen_at)
            VALUES (:u,:i,:p,:ls)
            ON CONFLICT(user_id,item_id) DO UPDATE SET
              preference=excluded.preference,
              last_seen_at=excluded.last_seen_at
            """,
            {
                "u": str(r["user_id"]),
                "i": str(r["item_id"]),
                "p": float(r["preference"]),
                "ls": r["last_seen_at"],
            },
        )
    return len(df_pref)


def ensure_schema():
    """Create/upgrade schema, idempotent."""
    sql = Path(__file__).with_name("models.sql").read_text(encoding="utf-8")
    log.debug("Ensuring schema")
    with engine().begin() as con:
        # Base DDL
        for stmt in sql.split(";"):
            s = stmt.strip()
            if s:
                try:
                    con.exec_driver_sql(s + ";")
                except Exception:
                    pass

        # Migrations handled here (not in models.sql):
        if not _column_exists(con, "users", "display_name"):
            log.info("Migrating: add users.display_name")
            con.exec_driver_sql("ALTER TABLE users ADD COLUMN display_name TEXT;")

        if not _column_exists(con, "watch_events", "plex_history_key"):
            log.info("Migrating: add watch_events.plex_history_key")
            con.exec_driver_sql("ALTER TABLE watch_events ADD COLUMN plex_history_key TEXT;")

        con.exec_driver_sql(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations(
              name TEXT PRIMARY KEY,
              applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
            )
            """
        )

        # Allow multiple managed/guest entries with empty username
        try:
            con.exec_driver_sql("UPDATE users SET user_name = NULL WHERE user_name = ''")
        except Exception:
            pass

        # feature columns on items (used by the vectorizer)
        item_migrations = [
            ("content_rating", "ALTER TABLE items ADD COLUMN content_rating TEXT"),
            ("countries_csv", "ALTER TABLE items ADD COLUMN countries_csv TEXT"),
            ("keywords_csv", "ALTER TABLE items ADD COLUMN keywords_csv TEXT"),
            ("tmdb_id", "ALTER TABLE items ADD COLUMN tmdb_id TEXT"),
        ]
        for col, ddl in item_migrations:
            if not _column_exists(con, "items", col):
                log.info("Migrating: add items.%s", col)
                con.exec_driver_sql(ddl)

        if not _migration_done(con, "watch_events_dedup_v1"):
            log.info("Running one-time watch_events dedup migration")
            before = con.exec_driver_sql("SELECT COUNT(*) FROM watch_events").scalar() or 0
            con.exec_driver_sql(
                """
                WITH ranked AS (
                  SELECT id,
                         ROW_NUMBER() OVER (
                           PARTITION BY COALESCE(NULLIF(plex_history_key, ''),
                             COALESCE(user_id, '') || '|' || COALESCE(item_id, '') || '|' || COALESCE(started_at, '') || '|' || COALESCE(source, '')
                           )
                           ORDER BY
                             CASE WHEN COALESCE(completed, 0) = 1 THEN 0 ELSE 1 END,
                             COALESCE(duration, 0) DESC,
                             COALESCE(view_offset, 0) DESC,
                             COALESCE(stopped_at, '') DESC,
                             id DESC
                         ) AS rn
                  FROM watch_events
                )
                DELETE FROM watch_events
                WHERE id IN (SELECT id FROM ranked WHERE rn > 1)
                """
            )
            after = con.exec_driver_sql("SELECT COUNT(*) FROM watch_events").scalar() or 0
            removed = int(before) - int(after)
            log.info("watch_events dedup removed %d duplicate rows", removed)

            con.exec_driver_sql(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS uq_we_source_history
                ON watch_events(source, plex_history_key)
                WHERE plex_history_key IS NOT NULL AND plex_history_key != ''
                """
            )
            con.exec_driver_sql(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS uq_we_fallback
                ON watch_events(user_id, item_id, started_at, source)
                """
            )

            pref_rows = _recompute_user_item_pref(con)
            log.info("Recomputed user_item_pref after dedup: %d rows", pref_rows)
            _mark_migration_done(con, "watch_events_dedup_v1")

        # Helpful indexes (idempotent; ignore errors if present)
        try:
            con.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_items_year ON items(year)")
            con.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_items_tmdb ON items(tmdb_id)")
            con.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_pref_user ON user_item_pref(user_id)")
        except Exception:
            pass

    log.debug("Schema ensured")
