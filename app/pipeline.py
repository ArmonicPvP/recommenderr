import math
import time
import logging
import datetime as dt
import pandas as pd
from .db import engine, ensure_schema
from .plex import iter_history, fetch_metadata, list_sections, iter_library_items, list_home_users
from .features import build_item_matrix
from .tmdb import fetch_keywords, parse_tmdb_id_from_guids

log = logging.getLogger(__name__)

def _ts_to_iso_utc(ts):
    if ts is None:
        return None
    try:
        sec = int(str(ts))
        return dt.datetime.utcfromtimestamp(sec).replace(tzinfo=dt.timezone.utc).isoformat().replace("+00:00", "Z")
    except Exception:
        return str(ts)

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

def _normalize_media_type(s: str | None) -> str | None:
    if not s:
        return None
    s = s.strip().lower()
    if s in ("movie", "film"):
        return "movie"
    if s in ("show", "tv", "series", "tv_show", "tvseries"):
        return "tv"
    return None

def _derive_media_type_from_md(md: dict) -> str | None:
    # Prefer explicit Plex "type"
    mt = _normalize_media_type(md.get("type"))
    if mt:
        return mt
    # Library/section hints
    mt = _normalize_media_type(md.get("library_section_type") or md.get("librarySectionType"))
    if mt:
        return mt
    # Some sources may have boolean flags
    if md.get("is_show") is True:
        return "tv"
    if md.get("is_movie") is True:
        return "movie"
    return None

def _derive_tmdb_id_from_md(md: dict) -> str | None:
    # If pre-parsed
    for k in ("tmdb_id", "tmdbId", "tmdbid"):
        v = str(md.get(k) or "").strip()
        if v.isdigit():
            return v
    # Parse from GUIDs
    for k in ("guids_csv", "guids", "guid_list", "guid"):
        v = md.get(k)
        if v:
            if isinstance(v, (list, tuple)):
                tid = parse_tmdb_id_from_guids(v)
            else:
                tid = parse_tmdb_id_from_guids(str(v))
            if tid:
                return tid
    return None

def _backfill_media_type_tmdb_and_keywords(batch_size: int = 500) -> int:
    """
    Fill missing/incorrect media_type ('movie'|'tv'), tmdb_id, and keywords_csv.
    - If row lacks media_type or tmdb_id, re-fetch Plex metadata for that item.
    - If tmdb_id exists but keywords_csv is empty, fetch TMDB keywords using the correct endpoint.
    Returns number of rows updated.
    """
    updated = 0
    from .config import TMDB_API_KEY

    with engine().begin() as con:
        rows = con.exec_driver_sql(
            """
            SELECT item_id,
                   COALESCE(media_type,'') AS media_type,
                   COALESCE(tmdb_id,'')    AS tmdb_id,
                   COALESCE(keywords_csv,'') AS keywords_csv
            FROM items
            WHERE
                (media_type IS NULL OR media_type = '')
             OR (tmdb_id   IS NULL OR tmdb_id   = '')
             OR ((keywords_csv IS NULL OR keywords_csv = '') AND (tmdb_id IS NOT NULL AND tmdb_id != ''))
            LIMIT :lim
            """,
            {"lim": batch_size},
        ).fetchall()

    if not rows:
        return 0

    for item_id, media_type, tmdb_id, keywords_csv in rows:
        mt = _normalize_media_type(media_type)  # may be None
        tid = str(tmdb_id).strip() or ""

        # If we are missing media_type or tmdb_id, re-fetch item metadata from Plex.
        if not mt or not tid:
            md = fetch_metadata(item_id) or {}
            mt = mt or _derive_media_type_from_md(md)
            tid = tid or (_derive_tmdb_id_from_md(md) or "")

            # If we derived something new, persist immediately to avoid re-doing work next pass.
            if (media_type or "") != (mt or "") or (tmdb_id or "") != (tid or ""):
                with engine().begin() as con:
                    con.exec_driver_sql(
                        """
                        UPDATE items
                        SET media_type = :mt,
                            tmdb_id    = :tid
                        WHERE item_id  = :iid
                        """,
                        {"mt": mt or None, "tid": tid or None, "iid": item_id},
                    )
                updated += 1

        # If we still don't have what we need, skip keywords
        new_keywords_csv = keywords_csv
        if TMDB_API_KEY and mt in ("movie", "tv") and tid.isdigit() and not keywords_csv:
            try:
                kws = fetch_keywords(mt, tid)
                if kws:
                    new_keywords_csv = ",".join(sorted(set(kws)))
            except Exception as e:
                log.warning("TMDB keywords failed for %s (%s:%s): %s", item_id, mt, tid, e)

        # Persist keywords if changed
        if (keywords_csv or "") != (new_keywords_csv or ""):
            with engine().begin() as con:
                con.exec_driver_sql(
                    "UPDATE items SET keywords_csv=:kw WHERE item_id=:iid",
                    {"kw": new_keywords_csv or None, "iid": item_id},
                )
            updated += 1

        # Be polite; TMDB mentions ~50 rps upper bounds
        time.sleep(0.01)

    if updated:
        log.info("Backfill media_type/tmdb_id/keywords: updated %d items", updated)
    return updated


# ---------------------------
# Pipelines
# ---------------------------

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

    # Fill media_type/tmdb_id/keywords for newly touched items
    try:
        _backfill_media_type_tmdb_and_keywords(batch_size=500)
    except Exception as e:
        log.warning("Backfill media/tmdb/keywords after ingest skipped: %s", e)

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

    # Fill media_type/tmdb_id/keywords for scanned items
    try:
        _backfill_media_type_tmdb_and_keywords(batch_size=1000)
    except Exception as e:
        log.warning("Backfill media/tmdb/keywords after scan skipped: %s", e)

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
    """Recompute user_item_pref and rebuild item vectors."""
    ensure_schema()

    with engine().begin() as con:
        # Build or refresh preferences from watch_events
        df_we = pd.read_sql_query("SELECT user_id,item_id,started_at,duration FROM watch_events", con)
        if not df_we.empty:
            rows = []
            for (u, i), grp in df_we.groupby(["user_id", "item_id"]):
                rows.append({
                    "user_id": u,
                    "item_id": i,
                    "preference": _preference(grp),
                    "last_seen_at": pd.to_datetime(grp["started_at"], utc=True, errors="coerce").max(),
                })
            df_pref = pd.DataFrame(rows)
            df_pref["last_seen_at"] = df_pref["last_seen_at"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            for _, r in df_pref.iterrows():
                con.exec_driver_sql(
                    """
                    INSERT INTO user_item_pref(user_id,item_id,preference,last_seen_at)
                    VALUES (:u,:i,:p,:t)
                    ON CONFLICT(user_id,item_id) DO UPDATE SET
                      preference=excluded.preference,
                      last_seen_at=excluded.last_seen_at
                    """,
                    {"u": r.user_id, "i": r.item_id, "p": float(r.preference), "t": r.last_seen_at},
                )

    # Ensure all referenced items exist, then backfill media/tmdb/keywords
    log.debug("Backfill check…")
    _backfill_missing_items()
    try:
        _backfill_media_type_tmdb_and_keywords(batch_size=1000)
    except Exception as e:
        log.warning("Backfill media/tmdb/keywords before build skipped: %s", e)

    # Pull items for feature building (include all columns features.py expects)
    with engine().begin() as con:
        items = pd.read_sql_query(
            """
            SELECT item_id, title, summary, year,
                   genres_csv, cast_csv, directors_csv, collections_csv,
                   countries_csv, content_rating, keywords_csv
            FROM items
            """,
            con,
        )

    if items.empty:
        log.info("Rebuild skipped: no items")
        return 0

    log.info("Rebuilding vectors for %d items…", len(items))
    build_item_matrix(items)
    return len(items)