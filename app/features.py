import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import normalize
from joblib import dump, load

from .config import (ART_DIR, FEAT_W_GENRES, FEAT_W_PEOPLE, FEAT_W_COLLECTIONS, FEAT_W_YEAR, YEAR_BUCKET_SIZE, YEAR_MIN, YEAR_MAX, FEAT_W_TITLE, FEAT_W_SUMMARY, FEAT_W_COUNTRY, FEAT_W_RATING, FEAT_W_KEYWORDS,
)

VEC_PATH = os.path.join(ART_DIR, "vectorizers.joblib")
MAT_PATH = os.path.join(ART_DIR, "items_X.npz")
IDX_PATH = os.path.join(ART_DIR, "items_index.csv")

def csv_tokenizer(s: str):
    """Split comma-separated fields into tokens."""
    if s is None:
        return []
    return [t.strip() for t in str(s).split(",") if t and t.strip()]

def _year_bucket_feature(year: pd.Series) -> pd.Series:
    y = pd.to_numeric(year, errors="coerce")
    mask = y.notna()
    y_clamped = y.clip(lower=YEAR_MIN, upper=YEAR_MAX)
    step = max(1, YEAR_BUCKET_SIZE)
    lo = (np.floor((y_clamped - YEAR_MIN) / step) * step + YEAR_MIN).astype("Int64")
    hi = (lo + step - 1).astype("Int64")
    labels = pd.Series(["year_unknown"] * len(year), index=year.index, dtype=object)
    if YEAR_BUCKET_SIZE == 10:
        labels[mask] = "year_" + lo[mask].astype(str).str[:4] + "s"
    else:
        labels[mask] = "year_" + lo[mask].astype(str) + "_" + hi[mask].astype(str)
    return labels.fillna("year_unknown")

def _l2_row_normalize(X: sp.csr_matrix) -> sp.csr_matrix:
    # safe L2 row-normalization for sparse matrices
    X = X.tocsr(copy=False)
    row_sums = np.sqrt(X.multiply(X).sum(axis=1)).A.ravel()
    row_sums[row_sums == 0.0] = 1.0
    inv = sp.diags(1.0 / row_sums)
    return inv.dot(X)

def build_item_matrix(items_df: pd.DataFrame):
    items_df = items_df.copy()
    items_df["item_id"] = items_df["item_id"].astype(str)

    # Ensure required columns exist
    for col in [
        "title", "summary", "genres_csv", "cast_csv", "directors_csv",
        "collections_csv", "year", "countries_csv", "content_rating",
        "keywords_csv", "media_type",
    ]:
        if col not in items_df.columns:
            items_df[col] = "" if col != "year" else None

    # --- Compute year buckets BEFORE vectorizing ---
    items_df["year_bucket"] = _year_bucket_feature(items_df["year"])

    # collections (binary/count on comma tokens)
    vec_collections = CountVectorizer(tokenizer=csv_tokenizer, lowercase=False)
    C = vec_collections.fit_transform(items_df["collections_csv"].fillna(""))

    # genres (TF-IDF on comma tokens)
    vec_genres = TfidfVectorizer(tokenizer=csv_tokenizer, lowercase=False, min_df=1)
    G = vec_genres.fit_transform(items_df["genres_csv"].fillna(""))


    # people (cast + directors) — TF-IDF on comma tokens
    vec_people = TfidfVectorizer(tokenizer=csv_tokenizer, lowercase=False, max_features=6000, min_df=2)
    P = vec_people.fit_transform((items_df["cast_csv"].fillna("") + "," + items_df["directors_csv"].fillna("")).str.strip(","))

    # title (TF-IDF; bigrams; low min_df)
    vec_title = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1)
    Tt = vec_title.fit_transform(items_df["title"].fillna(""))

    # summary (TF-IDF; bigrams; slightly higher min_df)
    vec_summary = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), min_df=2)
    Ts = vec_summary.fit_transform(items_df["summary"].fillna(""))

    # year buckets (Count)
    vec_year = CountVectorizer(token_pattern=r"[^,]+")
    Y = vec_year.fit_transform(items_df["year_bucket"].fillna("year_unknown"))

    # country (Count on comma tokens; supports multiple countries)
    vec_country = CountVectorizer(tokenizer=csv_tokenizer, lowercase=False)
    Co = vec_country.fit_transform(items_df["countries_csv"].fillna(""))

    # content rating (Count — usually a single token like "PG-13" or "TV-MA")
    vec_rating = CountVectorizer(token_pattern=r"[^,]+")
    R = vec_rating.fit_transform(items_df["content_rating"].fillna(""))

    # TMDB keywords (TF-IDF on comma tokens)
    vec_kw = TfidfVectorizer(tokenizer=csv_tokenizer, lowercase=False, min_df=1)
    K = vec_kw.fit_transform(items_df["keywords_csv"].fillna(""))

    # --- Weighted stack (order must match recommendations.py explain) ---
    mats, weights = [], []

    def add(m, w):
        if m is not None and m.shape[1] > 0 and w > 0:
            mats.append(m)
            weights.append(w)

    # Order: collections, genres, people, title, summary, year, country, rating, keywords
    add(C, FEAT_W_COLLECTIONS)
    add(G, FEAT_W_GENRES)
    add(P, FEAT_W_PEOPLE)
    add(Tt, FEAT_W_TITLE)
    add(Ts, FEAT_W_SUMMARY)
    add(Y, FEAT_W_YEAR)
    add(Co, FEAT_W_COUNTRY)
    add(R, FEAT_W_RATING)
    add(K, FEAT_W_KEYWORDS)

    if not mats:
        raise RuntimeError("No feature matrices produced; check inputs/weights")

    mats = [m.multiply(w) for m, w in zip(mats, weights)]
    X = sp.hstack(mats, format="csr")
    X = _l2_row_normalize(X)  # consistent with cosine scoring

    # Persist artifacts
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
    return X


def load_artifacts():
    vecs = load(VEC_PATH)
    X = sp.load_npz(MAT_PATH)
    id_index = pd.read_csv(IDX_PATH, dtype={"item_id": str})["item_id"].astype(str).tolist()
    return vecs, X, id_index