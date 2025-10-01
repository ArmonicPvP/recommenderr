import logging
import numpy as np
import pandas as pd
import scipy.sparse as sp
from typing import Tuple, List, Dict, Set, Optional

from .db import engine
from .features import load_artifacts
from .config import REC_COLLECTION_BOOST, REC_COLLECTION_LOOKBACK

log = logging.getLogger(__name__)


# ---------------------------
# User resolution / aliasing
# ---------------------------

def _get_user_row(query: str):
    with engine().begin() as con:
        row = con.exec_driver_sql(
            """
            SELECT user_id, user_name, display_name
            FROM users
            WHERE lower(COALESCE(user_name,'')) = lower(:q)
               OR lower(COALESCE(display_name,'')) = lower(:q)
               OR user_id = :q
            """,
            {"q": query},
        ).fetchone()
    return row

def _user_has_prefs(user_id: str) -> bool:
    with engine().begin() as con:
        hit = con.exec_driver_sql(
            "SELECT 1 FROM user_item_pref WHERE user_id=:u LIMIT 1", {"u": user_id}
        ).fetchone()
    return hit is not None

def _find_alias_with_prefs(user_row) -> Optional[str]:
    """
    Safe aliasing:
      - Prefer exact username match across rows (non-empty usernames only)
      - Optionally consider display_name match (non-empty, not 'guest'), still require candidate username present
      - Never alias to managed/guest rows (user_name IS NULL/empty)
      - No global 'only user with prefs' fallback (too risky)
    """
    if not user_row:
        return None

    uid, uname, dname = user_row[0], (user_row[1] or "").strip(), (user_row[2] or "").strip()

    if _user_has_prefs(uid):
        return uid

    def _first_candidate(sql: str, params: dict):
        with engine().begin() as con:
            rows = con.exec_driver_sql(sql, params).fetchall()
        for r in rows or []:
            cand_uid, cand_uname, _ = r[0], (r[1] or "").strip(), (r[2] or "").strip()
            # Skip managed/guest (empty username)
            if not cand_uname:
                continue
            if _user_has_prefs(cand_uid):
                log.info("Alias: using user_id=%s (has prefs) for requested uid=%s", cand_uid, uid)
                return cand_uid
        return None

    if uname:
        cand = _first_candidate(
            """
            SELECT user_id, user_name, display_name
            FROM users
            WHERE user_id != :uid
              AND lower(COALESCE(user_name,'')) = lower(:uname)
            """,
            {"uid": uid, "uname": uname},
        )
        if cand:
            return cand

    if dname and dname.lower() != "guest":
        cand = _first_candidate(
            """
            SELECT user_id, user_name, display_name
            FROM users
            WHERE user_id != :uid
              AND lower(COALESCE(display_name,'')) = lower(:dname)
            """,
            {"uid": uid, "dname": dname},
        )
        if cand:
            return cand

    log.debug(
        "Alias: no safe alias for uid=%s (uname=%r, dname=%r); will not cross-route",
        uid, uname, dname
    )
    return None

def _resolve_user(query: str) -> Tuple[Optional[str], Dict]:
    row = _get_user_row(query)
    if not row:
        return None, {}
    uid, uname, dname = row
    chosen = _find_alias_with_prefs(row) or uid
    info = {"user_name": uname, "display_name": dname, "resolved_from": uid, "chosen": chosen}
    if chosen != uid:
        log.info("Resolved user query=%r uid=%s → chosen uid=%s (alias with prefs)", query, uid, chosen)
    return chosen, info


# ---------------------------
# User vector construction
# ---------------------------

def _build_user_rowvec_from_pairs(pairs, X: sp.csr_matrix, id_index: List[str]) -> np.ndarray:
    if not pairs:
        return np.zeros((X.shape[1],), dtype=np.float32)
    idx = {iid: i for i, iid in enumerate(id_index)}
    rows, ts = [], []
    for iid, t in pairs:
        j = idx.get(str(iid))
        if j is not None:
            rows.append(j)
            ts.append(t)
    if not rows:
        return np.zeros((X.shape[1],), dtype=np.float32)
    W = X[rows]
    if any(ts):
        pts = pd.to_datetime(pd.Series(ts), utc=True, errors="coerce")
        if pts.notna().any():
            now = pts.max()
            ages = (now - pts.fillna(now)).dt.days.astype(float).to_numpy()
            w = np.exp(-ages / 90.0).reshape(-1, 1)
        else:
            w = np.ones((len(rows), 1), dtype=np.float32)
    else:
        w = np.ones((len(rows), 1), dtype=np.float32)
    v = (W.multiply(w)).sum(axis=0)
    v = np.asarray(v).ravel()
    n = np.linalg.norm(v)
    return v / (n + 1e-12)

def _user_rowvec(uid: str, X: sp.csr_matrix, id_index: List[str]) -> np.ndarray:
    with engine().begin() as con:
        dfp = pd.read_sql_query(
            """
            SELECT item_id, last_seen_at
            FROM user_item_pref
            WHERE user_id = :u
            ORDER BY last_seen_at DESC
            LIMIT 500
            """,
            con, params={"u": uid}
        )
    pairs = [(str(r.item_id), r.last_seen_at) for r in dfp.itertuples()] if not dfp.empty else []
    v = _build_user_rowvec_from_pairs(pairs, X, id_index)
    if v.any():
        return v
    with engine().begin() as con:
        dfw = pd.read_sql_query(
            """
            SELECT item_id, MAX(started_at) AS last_seen_at
            FROM watch_events
            WHERE user_id = :u
            GROUP BY item_id
            ORDER BY last_seen_at DESC
            LIMIT 500
            """,
            con, params={"u": uid}
        )
    pairs2 = [(str(r.item_id), r.last_seen_at) for r in dfw.itertuples()] if not dfw.empty else []
    return _build_user_rowvec_from_pairs(pairs2, X, id_index)


# ---------------------------
# Helpers for boosts/explain
# ---------------------------

def _parse_csv_set(val: str) -> Set[str]:
    if not val:
        return set()
    return {t.strip() for t in str(val).split(",") if t.strip()}

def _recent_user_collections(user_id: str, lookback: int) -> Set[str]:
    """Collect the set of collections from the user's most recent preferred items."""
    with engine().begin() as con:
        df = pd.read_sql_query(
            """
            SELECT p.item_id, i.collections_csv
            FROM user_item_pref p
            JOIN items i ON i.item_id = p.item_id
            WHERE p.user_id=:u
            ORDER BY p.last_seen_at DESC
            LIMIT :n
            """, con, params={"u": user_id, "n": lookback}
        )
    cols = set()
    if not df.empty:
        for s in df["collections_csv"].fillna(""):
            cols.update(_parse_csv_set(s))
    return cols

def _feature_blocks_from_vecs(vecs: dict) -> List[tuple[str, int]]:
    """
    Reconstruct the stacked block layout to support per-block explanations.
    Mirrors features.py (order and available vectorizers).
    """
    blocks: List[tuple[str, int]] = []

    def maybe_add(key: str, label: str):
        v = vecs.get(key)
        size = len(getattr(v, "vocabulary_", {}) or {}) if v is not None else 0
        if size > 0:
            blocks.append((label, size))

    # Order must match features.py stacking:
    # collections, genres, people, title, summary, year, country, rating, keywords
    maybe_add("vec_collections", "collections")
    maybe_add("vec_genres", "genres")
    maybe_add("vec_people", "people")
    maybe_add("vec_title", "title")
    maybe_add("vec_summary", "summary")
    maybe_add("vec_year", "year")
    maybe_add("vec_country", "country")
    maybe_add("vec_rating", "rating")
    maybe_add("vec_keywords", "keywords")
    return blocks

def _block_slices(blocks: List[tuple[str, int]]) -> List[tuple[str, slice]]:
    out: List[tuple[str, slice]] = []
    start = 0
    for name, size in blocks:
        stop = start + int(size)
        out.append((name, slice(start, stop)))
        start = stop
    return out


# ---------------------------
# Public API
# ---------------------------

def recommend_for_username(
    user_query: str,
    k: int = 10,
    explain: bool = False,
    media_type: Optional[str] = None,   # "movie", "tv", or None for all
):
    """Return top-K recommendations for a user (optionally restricted to media_type)."""
    media_type = (media_type or "").strip().lower() or None
    if media_type not in (None, "movie", "tv"):
        media_type = None  # be forgiving

    try:
        vecs, X, id_index = load_artifacts()
    except Exception as e:
        log.error("Artifacts not loaded: %s", e)
        return []

    uid, info = _resolve_user(user_query)
    if not uid:
        log.warning("User resolution failed for query=%r", user_query)
        return []

    with engine().begin() as con:
        df_w = pd.read_sql_query(
            "SELECT DISTINCT item_id FROM watch_events WHERE user_id=:u",
            con, params={"u": uid}
        )
        watched = set(df_w["item_id"].astype(str).tolist())

        # Pull minimal item columns + media_type for filtering
        df_items = pd.read_sql_query(
            """
            SELECT item_id, title, year, genres_csv, collections_csv, poster_url, media_type
            FROM items
            """,
            con
        )

    if df_w.empty:
        log.info(
            "User %s has no watch history yet (uid=%s)",
            info.get("user_name") or info.get("display_name") or user_query, uid
        )
    else:
        log.debug("User %s watched distinct items: %d", uid, len(watched))

    # Filter candidate ids by media_type if requested
    allowed_ids: Optional[Set[str]] = None
    if media_type:
        # Normalize to lower and handle NULLs
        mt = df_items["media_type"].fillna("").str.lower()
        allowed_ids = set(df_items.loc[mt == media_type, "item_id"].astype(str))

    # Build user preference vector
    uvec = _user_rowvec(uid, X, id_index)
    if not uvec.any():
        log.info("User %s has no preference vector yet (uid=%s)", user_query, uid)
        return []

    # Candidate rows (unwatched, and media-type constrained if provided)
    if allowed_ids is None:
        unwatched_idx = [i for i, iid in enumerate(id_index) if iid not in watched]
    else:
        unwatched_idx = [i for i, iid in enumerate(id_index) if (iid not in watched and iid in allowed_ids)]

    if not unwatched_idx:
        log.info("No unwatched candidates for uid=%s (media_type=%s)", uid, media_type or "any")
        return []

    # Scores from the item matrix vs user vector
    C = X[unwatched_idx]
    scores = np.asarray(C.dot(uvec)).ravel()

    # --- Collections rerank boost (binary) ---
    user_cols = _recent_user_collections(uid, REC_COLLECTION_LOOKBACK)
    if user_cols:
        # map candidate item_id -> collections set
        cand_ids = [id_index[i] for i in unwatched_idx]
        subset = df_items[df_items["item_id"].isin(cand_ids)][["item_id", "collections_csv"]].copy()
        subset["collections_set"] = subset["collections_csv"].fillna("").map(_parse_csv_set)
        col_map = dict(zip(subset["item_id"], subset["collections_set"]))

        boost_mask = np.array(
            [1.0 if (col_map.get(cid) and (col_map[cid] & user_cols)) else 0.0 for cid in cand_ids],
            dtype=np.float32,
        )

        scores = scores + REC_COLLECTION_BOOST * boost_mask
        boosted_set = {cid for cid, m in zip(cand_ids, boost_mask) if m > 0} if explain else set()
    else:
        boosted_set = set()

    # Rank & format
    def _percentile(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(x))
        return ranks / max(len(x) - 1, 1)

    pct = _percentile(scores)
    k = max(int(k), 1)
    topk_local = np.argsort(-scores)[:k]

    picks_idx = [unwatched_idx[i] for i in topk_local]
    picks_ids = [id_index[i] for i in picks_idx]
    picks_pct = [float(pct[i]) for i in topk_local]
    picks_scores = [float(scores[i]) for i in topk_local]

    ord_map = {iid: i for i, iid in enumerate(picks_ids)}
    view = (
        df_items[df_items["item_id"].isin(picks_ids)]
        .assign(__o=lambda d: d["item_id"].map(ord_map))
        .sort_values("__o")
    )

    pct_map = dict(zip(picks_ids, picks_pct))
    sc_map = dict(zip(picks_ids, picks_scores))
    max_pct_value = max(pct_map.values()) if pct_map else 0.0

    # explain blocks (match features.py)
    blocks_slices = None
    if explain:
        blocks_slices = _block_slices(_feature_blocks_from_vecs(vecs))

    results: List[Dict] = []
    for r in view.itertuples():
        pct_val = float(pct_map[r.item_id])
        match_pct = 100 if pct_val >= max_pct_value else int(np.floor(100.0 * pct_val))

        rec = {
            "item_id": r.item_id,
            "title": r.title,
            "year": r.year,
            "genres": (r.genres_csv or ""),
            "collections": (r.collections_csv or ""),
            "poster": r.poster_url,
            "media_type": (r.media_type or None),
            "relevancy": match_pct,
            "score": round(sc_map[r.item_id], 4),
        }

        if explain and blocks_slices:
            row_dense = X[picks_idx[ord_map[r.item_id]]].toarray().ravel()
            total = sc_map[r.item_id]
            eps = 1e-12
            expl = {}
            for name, sl in blocks_slices:
                c = float(np.dot(uvec[sl], row_dense[sl]))
                pct_contrib = (c / (total + eps)) * 100.0 if total > 0 else 0.0
                expl[name] = {"contribution": round(c, 6), "percent_of_score": round(pct_contrib, 2)}
            if r.item_id in boosted_set:
                # show boost as % of final score (post-boost), which avoids "None%"
                total_after = total + REC_COLLECTION_BOOST
                pct_boost = (REC_COLLECTION_BOOST / max(total_after, 1e-12)) * 100.0
                expl["collections_rerank_boost"] = {
                    "contribution": REC_COLLECTION_BOOST,
                    "percent_of_score": round(pct_boost, 2),
                }
            rec["explain"] = expl

        results.append(rec)

    if results:
        best = max(picks_scores) if picks_scores else float("nan")
        median = float(np.median(scores)) if scores.size else float("nan")
        log.info(
            "Recs for query=%r resolved uid=%s: candidates=%d topK=%d best_cos=%.4f median_cos=%.4f (media_type=%s)",
            user_query, uid, len(unwatched_idx), len(results), best, median, media_type or "any",
        )
    else:
        log.info("Recs for query=%r resolved uid=%s: no results (media_type=%s)", user_query, uid, media_type or "any")

    return results