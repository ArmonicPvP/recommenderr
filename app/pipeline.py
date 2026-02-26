import math
import time
import logging
import datetime as dt
import pandas as pd
from .db import engine, ensure_schema
from .plex import iter_history, fetch_metadata, list_sections, iter_library_items, list_home_users
from .features import build_item_matrix
from .tmdb import fetch_keywords_for_movie

def _enrich_keywords_if_possible():
    from .config import TMDB_API_KEY
    if not TMDB_API_KEY:
        return 0
    added = 0
    with engine().begin() as con:
        rows = con.exec_driver_sql(
            """
            SELECT item_id, tmdb_id FROM items
            WHERE (tmdb_id IS NOT NULL AND tmdb_id != '')
              AND (keywords_csv IS NULL OR keywords_csv = '')
            LIMIT 300
            """
        ).fetchall()
    for item_id, tmdb_id in rows:
        kw = fetch_keywords_for_movie(str(tmdb_id))
        if kw:
            csv = ",".join(sorted(set(kw)))
            with engine().begin() as con:
                con.exec_driver_sql(
                    "UPDATE items SET keywords_csv=:csv WHERE item_id=:iid",
                    {"csv": csv, "iid": str(item_id)}
                )
            added += 1
        time.sleep(0.02)  # ~50 rps max
    if added:
        log.info("TMDB keywords enriched for %d items", added)
    return added


log = logging.getLogger(__name__)

def _ts_to_iso_utc(ts):
    if ts is None:
        return None
    try:
        sec = int(str(ts))
        return dt.datetime.utcfromtimestamp(sec).replace(tzinfo=dt.timezone.utc).isoformat().replace("+00:00", "Z")
    except Exception:
        return str(ts)

PREF_WEIGHTS = {
    "w_amt": 0.55,
    "w_rec": 0.30,
    "w_dur": 0.10,
    "w_bias": 0.05,
}


def _compute_preferences(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out["event_count"] = pd.to_numeric(out["event_count"], errors="coerce").fillna(0)
    out["max_duration"] = pd.to_numeric(out["max_duration"], errors="coerce").fillna(0)

    last_seen = pd.to_datetime(out["last_seen_at"], utc=True, errors="coerce")
    now = pd.Timestamp.utcnow()
    days_since = (now - last_seen).dt.total_seconds().div(86400.0).fillna(999.0)

    recent = (-days_since / 90.0).map(math.exp)
    dur_score = (out["max_duration"] / 1800.0).clip(lower=0.0, upper=1.0)

    out["preference"] = (
        PREF_WEIGHTS["w_amt"] * out["event_count"].map(math.log1p)
        + PREF_WEIGHTS["w_rec"] * recent
        + PREF_WEIGHTS["w_dur"] * dur_score
        + PREF_WEIGHTS["w_bias"]
    )
    out["last_seen_at"] = last_seen.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return out


def ingest_once():
    """Sync users and ingest canonical watch history; upsert item metadata."""
    ensure_schema()
    t0 = time.time()
    with engine().begin() as con:
        # 1) Sync users and find admin uid
        user_map = {}
        admin_uid = None
        try:
            home_map = list_home_users()
            for uid, payload in home_map.items():
                uname = (payload.get("username") or "").strip() or None
                dname = (payload.get("display_name") or "").strip() or None
                is_admin = bool(payload.get("admin"))
                if is_admin:
                    admin_uid = uid

                con.exec_driver_sql(
                    """
                    INSERT INTO users(user_id, user_name, display_name)
                    VALUES (:uid, :uname, :dname)
                    ON CONFLICT(user_id) DO UPDATE SET
                      user_name=excluded.user_name,
                      display_name=excluded.display_name
                    """,
                    {"uid": uid, "uname": uname, "dname": dname},
                )
                user_map[uid] = {"username": uname, "display_name": dname, "admin": is_admin}
            log.info("User sync complete: %d users (admin_uid=%s)", len(home_map), admin_uid)
        except Exception as e:
            log.warning("Could not sync home users: %s", e)

        # 1a) If we know the admin, rewrite any PMS-local rows (user_id='1') to that real uid
        if admin_uid:
            updated_we = con.exec_driver_sql(
                "UPDATE watch_events SET user_id=:admin WHERE user_id='1'", {"admin": admin_uid}
            ).rowcount
            updated_pref = con.exec_driver_sql(
                "UPDATE user_item_pref SET user_id=:admin WHERE user_id='1'", {"admin": admin_uid}
            ).rowcount
            if updated_we or updated_pref:
                log.info("Remapped PMS user_id=1 → %s (we=%d, prefs=%d)", admin_uid, updated_we, updated_pref)

        # 2) Ingest history; also rewrite incoming '1' to admin_uid immediately
        count = 0
        for row in iter_history():
            raw_uid = str(row.get("user_id") or "")
            uid = admin_uid if (raw_uid == "1" and admin_uid) else raw_uid

            item_id = row.get("item_id")
            if not item_id:
                log.debug("Skipping history row without item_id: %s", row)
                continue

            cached = user_map.get(uid) or {}
            friendly = cached.get("username") or cached.get("display_name") or None

            started_iso = _ts_to_iso_utc(row.get("started_at"))
            stopped_iso = _ts_to_iso_utc(row.get("stopped_at"))

            con.exec_driver_sql(
                """
                INSERT INTO watch_events(
                    user_id, user_name, item_id,
                    started_at, stopped_at,
                    duration, view_offset, completed, source
                )
                VALUES (:uid, :uname, :iid, :st, :sp, :dur, :off, 1, 'plex')
                """,
                {
                    "uid": uid,
                    "uname": friendly,
                    "iid": item_id,
                    "st": started_iso,
                    "sp": stopped_iso,
                    "dur": row.get("duration") or 0,
                    "off": row.get("view_offset") or 0,
                },
            )

            md = fetch_metadata(item_id)
            if md:
                cols = ",".join(md.keys())
                placeholders = ",".join(":" + k for k in md.keys())
                sets = ",".join([f"{k}=excluded.{k}" for k in md.keys() if k != "item_id"])
                con.exec_driver_sql(
                    f"INSERT INTO items({cols}) VALUES ({placeholders}) "
                    f"ON CONFLICT(item_id) DO UPDATE SET {sets}",
                    md,
                )
            count += 1
    dt_ms = int((time.time() - t0) * 1000)
    log.info("Ingest complete: %d history rows in %dms", count, dt_ms)
    return count

def scan_library_all():
    """Upsert all movies/series from every movie/show section into items."""
    ensure_schema()
    t0 = time.time()
    sections = list_sections()
    total = 0
    with engine().begin() as con:
        for s in sections:
            kind = s.get("type")
            if kind not in ("movie", "show"):
                continue
            section_key = s["key"]
            for item_rk in iter_library_items(section_key, kind):
                md = fetch_metadata(item_rk)
                if not md:
                    continue
                cols = ",".join(md.keys())
                placeholders = ",".join(":" + k for k in md.keys())
                sets = ",".join([f"{k}=excluded.{k}" for k in md.keys() if k != "item_id"])
                con.exec_driver_sql(
                    f"INSERT INTO items({cols}) VALUES ({placeholders}) "
                    f"ON CONFLICT(item_id) DO UPDATE SET {sets}",
                    md,
                )
                total += 1
    dt_ms = int((time.time() - t0) * 1000)
    log.info("Library scan complete: %d items updated in %dms", total, dt_ms)
    return total

def _backfill_missing_items():
    """Ensure every referenced item exists in items."""
    filled = 0
    with engine().begin() as con:
        missing = con.exec_driver_sql(
            """
            WITH refs AS (
                SELECT DISTINCT item_id FROM user_item_pref
                UNION
                SELECT DISTINCT item_id FROM watch_events
            )
            SELECT r.item_id
            FROM refs r
            LEFT JOIN items i ON i.item_id = r.item_id
            WHERE i.item_id IS NULL
            """
        ).fetchall()
    missing_ids = [str(r[0]) for r in missing]
    if not missing_ids:
        log.debug("Backfill: no missing items")
        return 0
    log.info("Backfill: fetching metadata for %d missing items", len(missing_ids))
    with engine().begin() as con:
        for iid in missing_ids:
            md = fetch_metadata(iid)
            if not md:
                continue
            cols = ",".join(md.keys())
            placeholders = ",".join(":" + k for k in md.keys())
            sets = ",".join([f"{k}=excluded.{k}" for k in md.keys() if k != "item_id"])
            con.exec_driver_sql(
                f"INSERT INTO items({cols}) VALUES ({placeholders}) "
                f"ON CONFLICT(item_id) DO UPDATE SET {sets}",
                md,
            )
            filled += 1
    log.info("Backfill: added/updated %d items", filled)
    return filled

def rebuild_prefs_and_vectors():
    """Incrementally recompute user_item_pref and rebuild item vectors."""
    ensure_schema()

    with engine().begin() as con:
        try:
            _enrich_keywords_if_possible()
        except Exception as e:
            log.warning("Keyword enrichment skipped: %s", e)

        last_seen_id_raw = con.exec_driver_sql(
            "SELECT value FROM pipeline_state WHERE key='prefs_last_event_id'"
        ).scalar()
        try:
            last_seen_id = int(last_seen_id_raw) if last_seen_id_raw is not None else 0
        except Exception:
            last_seen_id = 0

        max_event_id = con.exec_driver_sql("SELECT COALESCE(MAX(id), 0) FROM watch_events").scalar() or 0

        changed_pref_rows = pd.DataFrame()
        if int(max_event_id) > last_seen_id:
            con.exec_driver_sql("DROP TABLE IF EXISTS tmp_pref_agg")
            con.exec_driver_sql(
                """
                CREATE TEMP TABLE tmp_pref_agg AS
                SELECT
                    user_id,
                    item_id,
                    COUNT(*) AS event_count,
                    MAX(COALESCE(duration, 0)) AS max_duration,
                    MAX(started_at) AS last_seen_at
                FROM watch_events
                WHERE id > :last_id
                GROUP BY user_id, item_id
                """,
                {"last_id": last_seen_id},
            )

            con.exec_driver_sql(
                """
                INSERT INTO user_item_pref(user_id, item_id, preference, last_seen_at, event_count, max_duration)
                SELECT user_id, item_id, 0.0, last_seen_at, event_count, max_duration
                FROM tmp_pref_agg
                ON CONFLICT(user_id,item_id) DO UPDATE SET
                  event_count=user_item_pref.event_count + excluded.event_count,
                  max_duration=MAX(user_item_pref.max_duration, excluded.max_duration),
                  last_seen_at=MAX(user_item_pref.last_seen_at, excluded.last_seen_at)
                """
            )

            changed_pref_rows = pd.read_sql_query(
                """
                SELECT p.user_id, p.item_id, p.event_count, p.max_duration, p.last_seen_at
                FROM user_item_pref p
                INNER JOIN tmp_pref_agg t
                  ON t.user_id = p.user_id AND t.item_id = p.item_id
                """,
                con,
            )

            con.exec_driver_sql(
                """
                INSERT INTO pipeline_state(key, value)
                VALUES ('prefs_last_event_id', :max_id)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """,
                {"max_id": str(max_event_id)},
            )
            con.exec_driver_sql("DROP TABLE IF EXISTS tmp_pref_agg")

        if not changed_pref_rows.empty:
            df_pref = _compute_preferences(changed_pref_rows)
            for _, r in df_pref.iterrows():
                con.exec_driver_sql(
                    """
                    INSERT INTO user_item_pref(user_id,item_id,preference,last_seen_at,event_count,max_duration)
                    VALUES (:u,:i,:p,:t,:ec,:md)
                    ON CONFLICT(user_id,item_id) DO UPDATE SET
                      preference=excluded.preference,
                      last_seen_at=excluded.last_seen_at,
                      event_count=excluded.event_count,
                      max_duration=excluded.max_duration
                    """,
                    {
                        "u": r.user_id,
                        "i": r.item_id,
                        "p": float(r.preference),
                        "t": r.last_seen_at,
                        "ec": int(r.event_count),
                        "md": int(r.max_duration),
                    },
                )

        items = pd.read_sql_query(
            """
            SELECT item_id, title, summary,
                   genres_csv, cast_csv, directors_csv,
                   collections_csv, year
            FROM items
            """,
            con,
        )

    if items.empty:
        log.info("Rebuild skipped: no items")
        return 0

    log.debug("Backfill check…")
    _backfill_missing_items()

    log.info("Rebuilding vectors for %d items…", len(items))
    build_item_matrix(items)
    return len(items)
