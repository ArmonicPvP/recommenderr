import httpx
import logging
from .config import TMDB_API_KEY

log = logging.getLogger(__name__)

def fetch_keywords_for_movie(tmdb_id: str) -> list[str]:
    if not TMDB_API_KEY or not tmdb_id:
        return []
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}/keywords"
    try:
        r = httpx.get(url, params={"api_key": TMDB_API_KEY}, timeout=15)
        if r.status_code == 429:
            log.warning("TMDB 429 rate-limited on id=%s", tmdb_id)
            return []
        r.raise_for_status()
        data = r.json()
        kws = data.get("keywords") or []
        return [k.get("name", "").strip() for k in kws if k.get("name")]
    except Exception as e:
        log.warning("TMDB keywords fetch failed for %s: %s", tmdb_id, e)
        return []
