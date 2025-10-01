from __future__ import annotations
import logging
import time
from typing import Iterable, Optional, Tuple
import httpx
from .config import TMDB_API_KEY

log = logging.getLogger(__name__)

TMDB_API = "https://api.themoviedb.org/3"

# A tiny, safe retry policy for 429 / transient network issues.
_MAX_RETRIES = 3
_BACKOFF_SECS = 1.5


def _get_json(path: str, params: dict | None = None, timeout: float = 10.0) -> Tuple[Optional[dict], int]:
    """
    Returns (json|None, status_code). Handles 429 with a small retry.
    """
    if not TMDB_API_KEY:
        return None, 401

    url = f"{TMDB_API}{path}"
    qp = {"api_key": TMDB_API_KEY}
    if params:
        qp.update(params)

    for attempt in range(_MAX_RETRIES):
        try:
            r = httpx.get(url, params=qp, timeout=timeout)
        except Exception as e:
            # network hiccup; retry a couple times
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_BACKOFF_SECS * (attempt + 1))
                continue
            log.warning("TMDB request failed (giving up): %s %s", url, e)
            return None, 0

        if r.status_code == 429 and attempt < _MAX_RETRIES - 1:
            # Respect Retry-After if present, else small backoff
            ra = r.headers.get("Retry-After")
            delay = float(ra) if ra and ra.isdigit() else _BACKOFF_SECS * (attempt + 1)
            log.warning("TMDB 429 rate-limited; retrying in %.1fs (path=%s)", delay, path)
            time.sleep(delay)
            continue

        if not r.is_success:
            # non-success (including final 429)
            code = r.status_code
            if code == 404:
                # Expected if we query the wrong media type (movie vs tv) or unknown ID
                log.debug("TMDB %s returned 404", path)
            else:
                log.warning("TMDB request failed (%s): %s", code, url)
            return None, code

        try:
            return r.json(), r.status_code
        except Exception as e:
            log.warning("TMDB JSON decode failed: %s (%s)", e, url)
            return None, r.status_code

    return None, 0  # should not reach


def fetch_keywords(media_type: str, tmdb_id: str | int) -> list[str]:
    """
    Fetch keyword names for a TMDB item.

    media_type: 'movie' or 'tv'
    tmdb_id: numeric id (str|int)
    """
    if not tmdb_id or media_type not in ("movie", "tv"):
        return []

    path = f"/{media_type}/{tmdb_id}/keywords"
    data, code = _get_json(path)

    if not data:
        return []

    # Movie payload: {"id": 2493, "keywords": [{id, name}, ...]}
    # TV payload:    {"id": 84958, "results":  [{id, name}, ...]}
    field = "keywords" if media_type == "movie" else "results"
    arr = data.get(field) or []
    return [k.get("name", "").strip() for k in arr if k.get("name")]


def fetch_keywords_for_movie(tmdb_id: str | int) -> list[str]:
    return fetch_keywords("movie", tmdb_id)


def fetch_keywords_for_tv(tmdb_id: str | int) -> list[str]:
    return fetch_keywords("tv", tmdb_id)


def parse_tmdb_id_from_guids(guids: Iterable[str] | str) -> Optional[str]:
    """
    Convenience helper for Plex metadata:
      Accepts a single GUID string or an iterable of GUID strings (e.g., ["imdb://...", "tmdb://2493"]).
      Returns the TMDB id as a string if found, else None.
    """
    if isinstance(guids, str):
        candidates = [guids]
    else:
        candidates = list(guids)

    for g in candidates:
        if not g:
            continue
        s = str(g)
        # Plex shows these like "tmdb://2493"
        if s.startswith("tmdb://"):
            tid = s.split("tmdb://", 1)[1].strip()
            return tid or None
    return None
