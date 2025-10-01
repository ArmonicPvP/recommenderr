import json
import logging
import typer

from .logger import init_logging
from .db import ensure_schema, engine
from .pipeline import ingest_once, rebuild_prefs_and_vectors
from .scheduler import start_scheduler
from .recommendations import recommend_for_username
from .config import PULL_INTERVAL_MIN

# init logging as early as possible
init_logging()
log = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)

@app.command()
def serve():
    """Run the background pipeline on an interval and stay alive."""
    ensure_schema()
    log.info("Starting scheduler loop (interval=%s min)", PULL_INTERVAL_MIN)
    start_scheduler(PULL_INTERVAL_MIN)
    import time
    while True:
        time.sleep(3600)

@app.command()
def ingest():
    """Pull Plex history and user mapping once."""
    ensure_schema()
    log.info("Manual ingest requested")
    n = ingest_once()
    log.info("Ingest complete: %d rows", n)
    typer.echo(f"Ingested {n} history rows")

@app.command()
def build():
    """Recompute preferences and rebuild item vectors."""
    ensure_schema()
    log.info("Manual build requested")
    n = rebuild_prefs_and_vectors()
    log.info("Build complete: ~%d items", n)
    typer.echo(f"Built vectors for ~{n} items")

@app.command()
def users():
    """List cached Plex users (username ↔ display name ↔ account ID)."""
    ensure_schema()
    with engine().begin() as con:
        rows = con.exec_driver_sql(
            """
            SELECT user_id, user_name, display_name
            FROM users
            ORDER BY
              CASE WHEN user_name IS NOT NULL AND user_name != '' THEN 0 ELSE 1 END,
              lower(COALESCE(display_name,'')), user_id
            """
        ).fetchall()

    if not rows:
        typer.echo(
            "No users found yet.\n"
            "Run: python -m app.cli ingest   (syncs plex.tv /api/home/users)\n"
        )
        return

    typer.echo("Known Plex users:")
    for uid, uname, dname in rows:
        if uname and dname and uname != dname:
            label = f"{uname} — {dname}"
        elif uname:
            label = uname
        else:
            label = f"(managed) {dname or 'unknown'}"
        typer.echo(f"  - {label}  [id={uid}]")

# ---- Positional-args friendly recommend command (no nargs, no strict flags) ----
@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def recommend(ctx: typer.Context):
    """
    Usage (positional only):
      python -m app.cli recommend "<username>" [k] [media] [json] [explain]

    Examples:
      python -m app.cli recommend "frontiergamers"
      python -m app.cli recommend "frontiergamers" 10
      python -m app.cli recommend "frontiergamers" 10 movie
      python -m app.cli recommend "frontiergamers" 10 tv json explain

    Notes:
      - media: 'movie' or 'tv' (omit for mixed).
      - 'json' switches output to JSON.
      - 'explain' includes feature-group contributions.
      - Unknown flags are ignored (so '--whatever' won't crash).
    """
    ensure_schema()

    args = list(ctx.args or [])
    if not args:
        typer.echo("Missing arguments. Example: python -m app.cli recommend \"frontiergamers\" 10 movie")
        raise typer.Exit(2)

    # Defaults
    username = args[0]
    k = 10
    media_type = None
    output = "text"
    explain = False

    # Tail to parse: [maybe-k] [maybe-media] [extras...]
    tail = args[1:]

    def _to_int(s):
        try:
            return int(s)
        except Exception:
            return None

    # If first tail token is an int => k
    if tail and _to_int(tail[0]) is not None:
        k = _to_int(tail[0])
        tail = tail[1:]

    # If next token looks like media => media_type
    if tail and tail[0].lower() in ("movie", "tv"):
        media_type = tail[0].lower()
        tail = tail[1:]

    # Parse remaining loose tokens (json/explain or best-effort flags)
    i = 0
    while i < len(tail):
        tok = tail[i].lower()
        if tok in ("json", "--json", "-j"):
            output = "json"
        elif tok in ("explain", "--explain"):
            explain = True
        elif tok.startswith("--media="):
            mt = tok.split("=", 1)[1].strip().lower()
            if mt in ("movie", "tv"):
                media_type = mt
        elif tok in ("--media", "-m"):
            # best-effort: consume next token as media if present
            if i + 1 < len(tail) and tail[i + 1].lower() in ("movie", "tv"):
                media_type = tail[i + 1].lower()
                i += 1
        elif tok.startswith("--k="):
            kk = _to_int(tok.split("=", 1)[1])
            if kk and kk > 0:
                k = kk
        elif tok in ("--k", "-k"):
            if i + 1 < len(tail):
                kk = _to_int(tail[i + 1])
                if kk and kk > 0:
                    k = kk
                i += 1
        elif tok.startswith("--output="):
            output = tok.split("=", 1)[1].strip().lower()
        elif tok in ("--output", "-o"):
            if i + 1 < len(tail):
                output = tail[i + 1].lower()
                i += 1
        # else: ignore unknown tokens/flags silently
        i += 1

    if not isinstance(k, int) or k < 1:
        k = 10
    if media_type not in (None, "movie", "tv"):
        typer.echo("Invalid media type. Use 'movie', 'tv', or omit.")
        raise typer.Exit(2)

    log.info(
        "Recommend requested: query=%r k=%d fmt=%s explain=%s media=%s",
        username, k, output, explain, media_type or "any"
    )

    recs = recommend_for_username(username, k=k, explain=explain, media_type=media_type)

    if not recs:
        log.warning("No recs available for query=%r", username)
        typer.echo(
            "No recommendations available.\n"
            "- Check users: python -m app.cli users\n"
            "- Ingest history: python -m app.cli ingest\n"
            "- Build vectors: python -m app.cli build"
        )
        return

    if output == "json":
        print(json.dumps(recs, ensure_ascii=False))
        return

    # Pretty text
    for r in recs:
        mt_tag = f"[{r.get('media_type','?').upper()}]" if r.get("media_type") else ""
        parts = [f"- {r['title']} ({r.get('year','')})", mt_tag, f"[{r.get('genres','-')}]"]
        if r.get("collections"):
            parts.append(f"<Collection: {r['collections']}>")
        parts.append(f"— {r['relevancy']}% match (cos={r['score']})")
        print("  ".join(p for p in parts if p))

        if explain and "explain" in r:
            xs = sorted(
                r["explain"].items(),
                key=lambda kv: -kv[1]["contribution"] if kv[1]["contribution"] is not None else -1e-12
            )
            def fmt_pct(v):
                return "n/a" if v is None else f"{v}%"
            why = "; ".join(f"{name}: {fmt_pct(d['percent_of_score'])} (c={d['contribution']})" for name, d in xs)
            print(f"    why → {why}")

@app.command()
def debug_user(query: str):
    """Diagnostic: resolve user, counts, index hits, and user-vector density."""
    from .recommendations import _resolve_user, _user_rowvec
    from .features import load_artifacts
    import pandas as pd
    import numpy as np

    ensure_schema()
    uid, info = _resolve_user(query)
    if not uid:
        typer.echo(f"Could not resolve user for: {query}")
        raise typer.Exit(1)

    label = info.get("user_name") or info.get("display_name") or "n/a"
    typer.echo(f"Resolved query='{query}' → user_id={uid} ({label})")

    with engine().begin() as con:
        we_cnt = con.exec_driver_sql("SELECT COUNT(*) FROM watch_events WHERE user_id=:u", {"u": uid}).scalar_one()
        pref_cnt = con.exec_driver_sql("SELECT COUNT(*) FROM user_item_pref WHERE user_id=:u", {"u": uid}).scalar_one()
        dfp = pd.read_sql_query(
            """
            SELECT item_id, last_seen_at
            FROM user_item_pref
            WHERE user_id=:u
            ORDER BY last_seen_at DESC
            LIMIT 20
            """,
            con, params={"u": uid}
        )

    typer.echo(f"watch_events rows: {we_cnt}")
    typer.echo(f"user_item_pref rows: {pref_cnt}")

    try:
        _, X, id_index = load_artifacts()
    except Exception as e:
        typer.echo(f"Artifacts not loaded: {e}")
        raise typer.Exit(1)

    index_set = set(id_index)
    if not dfp.empty:
        ids = [str(x) for x in dfp["item_id"].tolist()]
        hits = [i for i in ids if i in index_set]
        miss = [i for i in ids if i not in index_set]
        typer.echo(f"Recent pref item_ids (20): {ids}")
        typer.echo(f"In index: {len(hits)} / 20")
        if miss:
            typer.echo(f"Missing from index: {miss}")

    v = _user_rowvec(uid, X, id_index)
    nz = int(np.count_nonzero(v))
    typer.echo(f"user vector nonzeros: {nz} / {v.size}")

if __name__ == "__main__":
    log.info("CLI starting")
    app()