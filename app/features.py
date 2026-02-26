import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import normalize
from joblib import dump, load
from sklearn.neighbors import NearestNeighbors
from .config import (ART_DIR, FEAT_W_GENRES, FEAT_W_PEOPLE, FEAT_W_COLLECTIONS, FEAT_W_YEAR, YEAR_BUCKET_SIZE, YEAR_MIN, YEAR_MAX, FEAT_W_TITLE, FEAT_W_SUMMARY, FEAT_W_COUNTRY, FEAT_W_RATING, FEAT_W_KEYWORDS)

VEC_PATH = os.path.join(ART_DIR, "vectorizers.joblib")
MAT_PATH = os.path.join(ART_DIR, "items_X.npz")
IDX_PATH = os.path.join(ART_DIR, "items_index.csv")
NN_PATH = os.path.join(ART_DIR, "items_nn_index.joblib")
def _canonicalize_csv_field(series: pd.Series) -> pd.Series:
    """
    Normalize comma-separated string fields:
      - split on commas
      - trim whitespace
      - drop empties
      - de-duplicate while keeping stable order
      - join back with a single comma
    """
    def _dedup_keep_order(tokens):
        seen = set()
        out = []
        for t in tokens:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    series = series.fillna("").astype(str)
    cleaned = []
    for s in series:
        toks = [t.strip() for t in s.split(",") if t.strip()]
        toks = _dedup_keep_order(toks)
        cleaned.append(",".join(toks))
    return pd.Series(cleaned, index=series.index)


def _year_bucket_feature(year: pd.Series) -> pd.Series:
    """
    Map numeric year to buckets [lo..hi] with width YEAR_BUCKET_SIZE (>=1).
    Falls back to at least 1-year buckets defensively.
    """
    y = pd.to_numeric(year, errors="coerce")
    mask = y.notna()

    step = max(1, YEAR_BUCKET_SIZE)
    y_clamped = y.clip(lower=YEAR_MIN, upper=YEAR_MAX)

    lo = (np.floor((y_clamped - YEAR_MIN) / step) * step + YEAR_MIN).astype("Int64")
    hi = (lo + step - 1).astype("Int64")

    labels = pd.Series(["year_unknown"] * len(year), index=year.index, dtype=object)
    labels[mask] = "year_" + lo[mask].astype(str) + "_" + hi[mask].astype(str)
    return labels.fillna("year_unknown")


def _safe_fit_tf_idf_on_csv(vec: TfidfVectorizer, series: pd.Series) -> tuple[sp.csr_matrix, TfidfVectorizer | None]:
    """
    Fit a TF-IDF vectorizer on comma-separated fields safely.
    We pre-canonicalize to strip spaces so we can rely on token_pattern='[^,]+'.
    Returns (X, vec_or_None) where X may be empty (n x 0) if no tokens exist.
    """
    s = _canonicalize_csv_field(series)
    if (s == "").all():
        # all rows empty -> no vocab
        return sp.csr_matrix((len(s), 0), dtype=np.float32), None
    try:
        X = vec.fit_transform(s)
        return X, vec
    except ValueError:
        return sp.csr_matrix((len(s), 0), dtype=np.float32), None


def _safe_fit_count_on_csv(vec: CountVectorizer, series: pd.Series) -> tuple[sp.csr_matrix, CountVectorizer | None]:
    """
    Fit a Count vectorizer on comma-separated fields safely.
    """
    s = _canonicalize_csv_field(series)
    if (s == "").all():
        return sp.csr_matrix((len(s), 0), dtype=np.float32), None
    try:
        X = vec.fit_transform(s)
        return X, vec
    except ValueError:
        return sp.csr_matrix((len(s), 0), dtype=np.float32), None


def _safe_fit_text(vec, series: pd.Series) -> tuple[sp.csr_matrix, object | None]:
    """
    Fit a word/bigram text vectorizer on free text safely.
    """
    s = series.fillna("").astype(str)
    if (s.str.strip() == "").all():
        return sp.csr_matrix((len(s), 0), dtype=np.float32), None
    try:
        X = vec.fit_transform(s)
        return X, vec
    except ValueError:
        return sp.csr_matrix((len(s), 0), dtype=np.float32), None


# ---------- main builder ----------

def build_item_matrix(items_df: pd.DataFrame):
    items_df = items_df.copy()
    items_df["item_id"] = items_df["item_id"].astype(str)

    # Ensure required columns exist
    for col in [
        "title", "summary", "genres_csv", "cast_csv", "directors_csv",
        "collections_csv", "year", "countries_csv", "content_rating", "keywords_csv"
    ]:
        if col not in items_df.columns:
            items_df[col] = "" if col != "year" else None

    # collections (binary-ish, strong)
    vec_collections = CountVectorizer(token_pattern=r"[^,]+", lowercase=False)
    C, vec_collections = _safe_fit_count_on_csv(vec_collections, items_df["collections_csv"])

    # genres (TF-IDF on comma tokens)
    vec_genres = TfidfVectorizer(token_pattern=r"[^,]+", lowercase=False, min_df=1)
    G, vec_genres = _safe_fit_tf_idf_on_csv(vec_genres, items_df["genres_csv"])

    # people (cast + directors) as TF-IDF on comma tokens
    people_csv = (items_df["cast_csv"].fillna("") + "," + items_df["directors_csv"].fillna("")).str.strip(",")
    vec_people = TfidfVectorizer(token_pattern=r"[^,]+", lowercase=False, max_features=6000, min_df=2)
    P, vec_people = _safe_fit_tf_idf_on_csv(vec_people, people_csv)

    # title (TF-IDF; bigrams; low min_df to keep distinctive titles)
    vec_title = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1)
    Tt, vec_title = _safe_fit_text(vec_title, items_df["title"])

    # summary (TF-IDF; bigrams; slightly higher min_df to reduce noise)
    vec_summary = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), min_df=2)
    Ts, vec_summary = _safe_fit_text(vec_summary, items_df["summary"])

    # year buckets (Count)
    items_df["year_bucket"] = _year_bucket_feature(items_df["year"])
    vec_year = CountVectorizer(token_pattern=r"[^,]+", lowercase=False)
    Y, vec_year = _safe_fit_count_on_csv(vec_year, items_df["year_bucket"])

    # country (Count on comma tokens) — countries_csv may be multi-valued
    vec_country = CountVectorizer(token_pattern=r"[^,]+", lowercase=False)
    Co, vec_country = _safe_fit_count_on_csv(vec_country, items_df["countries_csv"])

    # content rating (Count, single token per row typical)
    vec_rating = CountVectorizer(token_pattern=r"[^,]+", lowercase=False)
    R, vec_rating = _safe_fit_count_on_csv(vec_rating, items_df["content_rating"])

    # TMDB keywords (TF-IDF on comma tokens)
    vec_kw = TfidfVectorizer(token_pattern=r"[^,]+", lowercase=False, min_df=1)
    K, vec_kw = _safe_fit_tf_idf_on_csv(vec_kw, items_df["keywords_csv"])

    # Weighted stack
    mats, weights = [], []

    def add(m, w):
        if m is not None and m.shape[1] > 0 and w > 0:
            mats.append(m)
            weights.append(w)

    add(C,  FEAT_W_COLLECTIONS)
    add(G,  FEAT_W_GENRES)
    add(P,  FEAT_W_PEOPLE)
    add(Tt, FEAT_W_TITLE)
    add(Ts, FEAT_W_SUMMARY)
    add(Y,  FEAT_W_YEAR)
    add(Co, FEAT_W_COUNTRY)
    add(R,  FEAT_W_RATING)
    add(K,  FEAT_W_KEYWORDS)

    if not mats:
        raise RuntimeError("No feature matrices produced; check inputs/weights and source data.")

    # Apply weights, stack, normalize
    mats = [m.multiply(w) for m, w in zip(mats, weights)]
    X = sp.hstack(mats, format="csr")
    X = normalize(X)

    dump(
        {
            "vec_collections": vec_collections,
            "vec_genres": vec_genres,
            "vec_people": vec_people,
            "vec_title": vec_title,
            "vec_summary": vec_summary,
            "vec_year": vec_year,
            "vec_country": vec_country,
            "vec_rating": vec_rating,
            "vec_keywords": vec_kw,
            "weights": {
                "collections": FEAT_W_COLLECTIONS,
                "genres": FEAT_W_GENRES,
                "people": FEAT_W_PEOPLE,
                "title": FEAT_W_TITLE,
                "summary": FEAT_W_SUMMARY,
                "year": FEAT_W_YEAR,
                "country": FEAT_W_COUNTRY,
                "rating": FEAT_W_RATING,
                "keywords": FEAT_W_KEYWORDS,
                "year_bucket_size": YEAR_BUCKET_SIZE,
            },
        },
        VEC_PATH,
    )

    sp.save_npz(MAT_PATH, X)
    items_df[["item_id"]].astype(str).to_csv(IDX_PATH, index=False)

    nn_index = None
    if X.shape[0] > 0:
        nn_index = NearestNeighbors(metric="cosine", algorithm="brute")
        nn_index.fit(X)
    dump({"nn_index": nn_index, "metric": "cosine"}, NN_PATH)
    return X


def load_artifacts():
    vecs = load(VEC_PATH)
    X = sp.load_npz(MAT_PATH)
    id_index = pd.read_csv(IDX_PATH, dtype={"item_id": str})["item_id"].astype(str).tolist()

    nn_artifact = None
    if os.path.exists(NN_PATH):
        nn_artifact = load(NN_PATH)
    return vecs, X, id_index, nn_artifact
