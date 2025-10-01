import logging
import httpx
import xmltodict
from .config import PLEX_BASE, PLEX_TOKEN

log = logging.getLogger(__name__)

def _get(path: str, params=None) -> str:
    params = params or {}
    params["X-Plex-Token"] = PLEX_TOKEN
    url = f"{PLEX_BASE}{path}"
    log.debug("GET %s params=%s", url, {k: v for k, v in params.items() if k != "X-Plex-Token"})
    r = httpx.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.text

def _extract_id_from_keypath(path: str | None) -> str | None:
    if not path:
        return None
    if "/library/metadata/" in path:
        tail = path.split("/library/metadata/", 1)[-1]
        return tail.split("?", 1)[0].strip("/ ")
    return None

def iter_history():
    """Yield canonicalized watch events (episodes → series id; movies unchanged)."""
    xml = _get("/status/sessions/history/all")
    root = xmltodict.parse(xml).get("MediaContainer", {})
    videos = root.get("Video")
    if videos is None:
        log.info("History: 0 rows")
        return
    if isinstance(videos, dict):
        videos = [videos]
    log.info("History: %d rows (raw)", len(videos))

    kept = 0
    for v in videos:
        media_type = v.get("@type")
        rk = v.get("@ratingKey") or _extract_id_from_keypath(v.get("@key"))
        if media_type == "episode":
            series_rk = v.get("@grandparentRatingKey") or _extract_id_from_keypath(v.get("@grandparentKey"))
            canonical_id = str(series_rk) if series_rk else (str(rk) if rk else None)
        else:
            canonical_id = str(rk) if rk else None
        if not canonical_id:
            continue
        kept += 1
        yield {
            "user_id": str(v.get("@accountID") or ""),
            "user_name": None,
            "item_id": canonical_id,
            "type": "show" if media_type == "episode" else media_type,
            "title": v.get("@title"),
            "started_at": v.get("@viewedAt") or v.get("@addedAt"),
            "stopped_at": v.get("@viewedAt"),
            "duration": int(v.get("@duration", 0)) // 1000 if v.get("@duration") else 0,
            "view_offset": int(v.get("@viewOffset", 0)) // 1000 if v.get("@viewOffset") else 0,
        }
    log.info("History: %d rows after canonicalization", kept)

def fetch_metadata(item_id: str):
    """Fetch metadata for a movie or series; return fields used for vectorization."""
    if not item_id or str(item_id).lower() == "none":
        return None

    xml = _get(f"/library/metadata/{item_id}")
    mc = xmltodict.parse(xml).get("MediaContainer", {})
    node = mc.get("Video") or mc.get("Directory")
    if isinstance(node, list):
        node = node[0]
    if not node:
        log.warning("No metadata node for item_id=%s", item_id)
        return None

    def csv_attr(node, key):
        xs = node.get(key)
        if not xs:
            return ""
        if isinstance(xs, list):
            return ",".join([x.get("@tag", "") for x in xs if isinstance(x, dict)])
        if isinstance(xs, dict):
            return xs.get("@tag", "")
        return ""

    roles = node.get("Role")
    cast = ""
    if roles:
        if isinstance(roles, list):
            cast = ",".join([r.get("@tag", "") for r in roles if isinstance(r, dict)])
        elif isinstance(roles, dict):
            cast = roles.get("@tag", "")

    genres = csv_attr(node, "Genre")
    directors = csv_attr(node, "Director")
    collections = csv_attr(node, "Collection")
    countries = csv_attr(node, "Country")
    content_rating = node.get("@contentRating")
    thumb = node.get("@thumb")
    poster_url = f"{PLEX_BASE}{thumb}?X-Plex-Token={PLEX_TOKEN}" if thumb else None
    runtime = int(node.get("@duration", 0)) // 1000 if node.get("@duration") else None
    year = node.get("@year")
    year = int(year) if year and str(year).isdigit() else None

    tmdb_id = None
    guids = node.get("Guid")
    if guids:
        if isinstance(guids, dict):
            guids = [guids]
        for g in guids:
            gid = g.get("@id") if isinstance(g, dict) else None
            if gid and gid.startswith("tmdb://"):
                tmdb_id = gid.split("tmdb://", 1)[1].strip()
                break

    md = {
        "item_id": str(node.get("@ratingKey") or item_id),
        "type": node.get("@type"),
        "title": node.get("@title"),
        "year": year,
        "runtime": runtime,
        "summary": node.get("@summary"),
        "genres_csv": genres,
        "cast_csv": cast,
        "directors_csv": directors,
        "collections_csv": collections,
        "poster_url": poster_url,
        "countries_csv": countries or "",
        "content_rating": content_rating or "",
        "tmdb_id": tmdb_id or ""
    }
    log.debug("Fetched metadata: %s", {"item_id": md["item_id"], "type": md["type"], "title": md["title"]})
    return md

def list_sections():
    """Return all library sections with basic info."""
    xml = _get("/library/sections")
    mc = xmltodict.parse(xml).get("MediaContainer", {})
    dirs = mc.get("Directory") or []
    if isinstance(dirs, dict):
        dirs = [dirs]
    out = [{"key": str(d.get("@key")), "title": d.get("@title"), "type": d.get("@type")} for d in dirs]
    log.info("Sections: %s", [{"key": d["key"], "title": d["title"], "type": d["type"]} for d in out])
    return out

def iter_library_items(section_key: str, kind: str):
    """Yield ratingKeys for top-level items in a section (movie/show)."""
    xml = _get(f"/library/sections/{section_key}/all")
    mc = xmltodict.parse(xml).get("MediaContainer", {})
    nodes = []
    if mc.get("Video"):
        nodes.extend(mc["Video"] if isinstance(mc["Video"], list) else [mc["Video"]])
    if mc.get("Directory"):
        nodes.extend(mc["Directory"] if isinstance(mc["Directory"], list) else [mc["Directory"]])
    if not nodes:
        log.info("Section %s: 0 nodes", section_key)
        return
    want = "movie" if kind == "movie" else "show"
    cnt = 0
    for n in nodes:
        ntype = n.get("@type") or n.get("type")
        if ntype != want:
            continue
        rk = n.get("@ratingKey") or _extract_id_from_keypath(n.get("@key"))
        if rk:
            cnt += 1
            yield str(rk)
    log.info("Section %s (%s): yielded %d items", section_key, want, cnt)

def list_home_users():
    """Map plex.tv Home users using the same PLEX_TOKEN → {id: {username, display_name}}."""
    url = "https://plex.tv/api/home/users"
    params = {"X-Plex-Token": PLEX_TOKEN}
    log.debug("GET %s params=%s", url, {"X-Plex-Token": "***"})
    r = httpx.get(url, params=params, timeout=20)
    r.raise_for_status()
    mc = xmltodict.parse(r.text).get("MediaContainer", {})
    users = mc.get("User") or []
    if isinstance(users, dict):
        users = [users]
    out = {}
    for u in users:
        uid = str(u.get("@id"))
        username = (u.get("@username") or "").strip()
        title = (u.get("@title") or "").strip()
        admin = str(u.get("@admin", "0")) == "1"
        if uid:
            out[uid] = {"username": username, "display_name": title, "admin": admin}
    log.info("Home users mapped: %d", len(out))
    return out