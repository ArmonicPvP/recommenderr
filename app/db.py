import logging
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from pathlib import Path
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

        # Allow multiple managed/guest entries with empty username
        try:
            con.exec_driver_sql("UPDATE users SET user_name = NULL WHERE user_name = ''")
        except Exception:
            pass

    log.debug("Schema ensured")