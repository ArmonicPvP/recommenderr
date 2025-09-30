# app/recommendations.py
import logging
import numpy as np
import pandas as pd
import scipy.sparse as sp
from .db import engine
from .features import load_artifacts

log = logging.getLogger(__name__)

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

def _find_alias_with_prefs(user_row) -> str | None:
    if not user_row:
        return None
    uid, uname, dname = user_row[0], user_row[1], user_row[2]
    if _user_has_prefs(uid):
        return uid
    with engine().begin() as con:
        alias = con.exec_driver_sql(
            """
            WITH target AS (
              SELECT lower(COALESCE(:uname,'')) AS u, lower(COALESCE(:dname,'')) AS d
            )
            SELECT u2.user_id
            FROM users u1
            CROSS JOIN target t
            JOIN users u2
              ON (
                    (lower(COALESCE(u1.user_name,''))   = t.u AND t.u != '')
                 OR (lower(COALESCE(u1.display_name,''))= t.d AND t.d != '')
                 OR (lower(COALESCE(u1.user_name,''))   = t.d AND t.d != '')
                 OR (lower(COALESCE(u1.display_name,''))= t.u AND t.u != '')
                 )
            WHERE u1.user_id = :uid AND u2.user_id != :uid
            """,
            {"uid": uid, "uname": uname or "", "dname": dname or ""},
        ).fetchall()
    for r in alias or []:
        cand = r[0]
        if _user_has_prefs(cand):
            log.info("Alias: using user_id=%s (has prefs) for requested uid=%s", cand, uid)
            return cand
    with engine().begin() as con:
        pref_holders = con.exec_driver_sql(
            "SELECT user_id, COUNT(*) c FROM user_item_pref GROUP BY user_id ORDER BY c DESC"
        ).fetchall()
    if len(pref_holders) == 1:
        only_uid = pref_holders[0][0]
        log.info("Alias fallback: using the only user_id with prefs: %s", only_uid)
        return only_uid
    return None

def _resolve_user(query: str) -> tuple[str | None, dict]:
    row = _get_user_row(query)
    if not row:
        return None, {}
    uid, uname, dname = row
    chosen = _find_alias_with_prefs(row) or uid
    info = {"user_name": uname, "display_name": dname, "resolved_from": uid, "chosen": chosen}
    if chosen != uid:
        log.info("Resolved user query=%r uid=%s → chosen uid=%s (alias with prefs)", query, uid, chosen)
    return chosen, info

def _build_user_rowvec_from_pairs(pairs, X: sp.csr_matrix, id_index: list[str]) -> np.ndarray:
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

def _user_rowvec(uid: str, X: sp.csr_matrix, id_index: list[str]) -> np.ndarray:
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

def _percentile(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(scores))
    return ranks / max(len(scores) - 1, 1)

def _feature_blocks(vecs: dict) -> list[tuple[str, int]]:
    blocks = []
    def maybe_add(name_key, weight_key):
        v = vecs.get(name_key)
        w = vecs.get("weights", {}).get(weight_key, 1.0)
        size = len(getattr(v, "vocabulary_", {}) or {}) if v is not None else 0
        if v is not None and w > 0 and size > 0:
            blocks.append((weight_key if weight_key != "text" else "text", size))
    maybe_add("vec_collections", "collections")
    maybe_add("vec_genres", "genres")
    maybe_add("vec_people", "people")
    maybe_add("vec_text", "text")
    maybe_add("vec_year", "year")
    return blocks

def _block_slices(blocks: list[tuple[str, int]]) -> list[tuple[str, slice]]:
    out, start = [], 0
    for name, size in blocks:
        stop = start + int(size)
        out.append((name, slice(start, stop)))
        start = stop
    return out

def _explain_item(uvec: np.ndarray, item_row_dense: np.ndarray, blocks_slices: list[tuple[str, slice]], total_score: float):
    expl, eps = {}, 1e-12
    for name, sl in blocks_slices:
        c = float(np.dot(uvec[sl], item_row_dense[sl]))
        pct = (c / (total_score + eps)) * 100.0 if total_score > 0 else 0.0
        expl[name] = {"contribution": round(c, 6), "percent_of_score": round(pct, 2)}
    return expl

def recommend_for_username(user_query: str, k: int = 10, explain: bool = False):
    """Return top-K recommendations for a user."""
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
        df_items = pd.read_sql_query(
            "SELECT item_id, title, year, genres_csv, collections_csv, poster_url FROM items",
            con
        )

    if df_w.empty:
        log.info("User %s has no watch history yet (uid=%s)",
                 info.get("user_name") or info.get("display_name") or user_query, uid)
    else:
        log.debug("User %s watched distinct items: %d", uid, len(watched))

    uvec = _user_rowvec(uid, X, id_index)
    if not uvec.any():
        log.info("User %s has no preference vector yet (uid=%s)", user_query, uid)
        return []

    unwatched_idx = [i for i, iid in enumerate(id_index) if iid not in watched]
    if not unwatched_idx:
        log.info("No unwatched candidates for uid=%s", uid)
        return []

    C = X[unwatched_idx]
    scores = np.asarray(C.dot(uvec)).ravel()

    pct = _percentile(scores)
    k = max(int(k), 1)
    topk_local = np.argsort(-scores)[:k]
    picks_idx = [unwatched_idx[i] for i in topk_local]
    picks_ids = [id_index[i] for i in picks_idx]
    picks_pct = [float(pct[i]) for i in topk_local]
    picks_scores = [float(scores[i]) for i in topk_local]

    ord_map = {iid: i for i, iid in enumerate(picks_ids)}
    out = (df_items[df_items["item_id"].isin(picks_ids)]
           .assign(__o=lambda d: d["item_id"].map(ord_map))
           .sort_values("__o"))

    pct_map = dict(zip(picks_ids, picks_pct))
    sc_map  = dict(zip(picks_ids, picks_scores))
    max_pct_value = max(pct_map.values()) if pct_map else 0.0
    blocks_slices = _block_slices(_feature_blocks(vecs)) if explain else None

    results = []
    for r in out.itertuples():
        pct_val = float(pct_map[r.item_id])
        match_pct = 100 if pct_val >= max_pct_value else int(np.floor(100.0 * pct_val))
        rec = {
            "item_id": r.item_id,
            "title": r.title,
            "year": r.year,
            "genres": (r.genres_csv or ""),
            "collections": (r.collections_csv or ""),
            "poster": r.poster_url,
            "relevancy": match_pct,
            "score": round(sc_map[r.item_id], 4)
        }
        if explain and blocks_slices:
            pos = ord_map[r.item_id]
            row_idx = picks_idx[pos]
            row_dense = X[row_idx].toarray().ravel()
            rec["explain"] = _explain_item(uvec, row_dense, blocks_slices, sc_map[r.item_id])
        results.append(rec)

    if results:
        best = max(picks_scores) if picks_scores else float("nan")
        median = float(np.median(scores)) if scores.size else float("nan")
        log.info("Recs for query=%r resolved uid=%s: candidates=%d topK=%d best_cos=%.4f median_cos=%.4f",
                 user_query, uid, len(unwatched_idx), len(results), best, median)
    else:
        log.info("Recs for query=%r resolved uid=%s: no results", user_query, uid)

    return results