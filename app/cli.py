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

@app.command()
def recommend(
    username: str = typer.Argument(..., help="Username, display name, or account ID"),
    k_arg: int = typer.Argument(10, min=1, help="How many recommendations (positional)"),
    k_opt: int = typer.Option(None, "--k", "-k", min=1, help="How many recommendations (option)"),
    output: str = typer.Option("text", "--output", "-o", help="text or json"),
    explain: bool = typer.Option(False, "--explain", help="Show feature-group contributions"),
    include_auth_token: bool = typer.Option(False, "--include-auth-token", help="Emit transient signed poster URLs with X-Plex-Token"),
):
    """Print top-K recommendations for a user."""
    ensure_schema()
    k = k_opt if k_opt is not None else k_arg

    log.info("Recommend requested: query=%r k=%d fmt=%s explain=%s", username, k, output, explain)
    recs = recommend_for_username(username, k=k, explain=explain, include_auth_token=include_auth_token)

    if not recs:
        log.warning("No recs available for query=%r", username)
        typer.echo(
            "No recommendations available.\n"
            "- Check users: python -m app.cli users\n"
            "- Ingest history: python -m app.cli ingest\n"
            "- Build vectors: python -m app.cli build"
        )
        return

    if output.lower() == "json":
        print(json.dumps(recs, ensure_ascii=False))
        return

    # Pretty text
    for r in recs:
        parts = [f"- {r['title']} ({r.get('year','')})", f"[{r.get('genres','-')}]"]
        if r.get("collections"):
            parts.append(f"<Collection: {r['collections']}>")
        parts.append(f"— {r['relevancy']}% match (cos={r['score']})")
        print("  ".join(parts))
        if explain and "explain" in r:
            xs = sorted(r["explain"].items(), key=lambda kv: -kv[1]["contribution"])
            why = "; ".join(f"{name}: {d['percent_of_score']}% (c={d['contribution']})" for name, d in xs)
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