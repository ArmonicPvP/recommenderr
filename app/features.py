import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import normalize
from joblib import dump, load
from .config import (
    ART_DIR,
    FEAT_W_GENRES, FEAT_W_PEOPLE, FEAT_W_TEXT, FEAT_W_COLLECTIONS, FEAT_W_YEAR,
    YEAR_BUCKET_SIZE, YEAR_MIN, YEAR_MAX,
)

VEC_PATH = os.path.join(ART_DIR, "vectorizers.joblib")
MAT_PATH = os.path.join(ART_DIR, "items_X.npz")
IDX_PATH = os.path.join(ART_DIR, "items_index.csv")

def _year_bucket_feature(year: pd.Series) -> pd.Series:
    y = pd.to_numeric(year, errors="coerce")
    mask = y.notna()
    y_clamped = y.clip(lower=YEAR_MIN, upper=YEAR_MAX)
    lo = (np.floor((y_clamped - YEAR_MIN) / max(1, YEAR_BUCKET_SIZE)) * YEAR_BUCKET_SIZE + YEAR_MIN).astype("Int64")
    hi = (lo + YEAR_BUCKET_SIZE - 1).astype("Int64")
    labels = pd.Series(["year_unknown"] * len(year), index=year.index, dtype=object)
    if YEAR_BUCKET_SIZE == 10:
        labels[mask] = "year_" + lo[mask].astype(str).str[:4] + "s"
    else:
        labels[mask] = "year_" + lo[mask].astype(str) + "_" + hi[mask].astype(str)
    return labels.fillna("year_unknown")

def build_item_matrix(items_df: pd.DataFrame):
    items_df = items_df.copy()
    items_df["item_id"] = items_df["item_id"].astype(str)

    # ensure required columns exist
    for col in ["title", "summary", "genres_csv", "cast_csv", "directors_csv", "collections_csv", "year"]:
        if col not in items_df.columns:
            items_df[col] = None if col == "year" else ""

    # text
    items_df["text"] = (items_df["title"].fillna("") + ". " + items_df["summary"].fillna("")).str.strip()

    # collections
    vec_collections = CountVectorizer(token_pattern=r"[^,]+")
    C = vec_collections.fit_transform(items_df["collections_csv"].fillna(""))

    # genres
    vec_genres = TfidfVectorizer(token_pattern=r"[^,]+", use_idf=True, norm=None)
    G = vec_genres.fit_transform(items_df["genres_csv"].fillna(""))

    # cast + directors
    vec_people = CountVectorizer(token_pattern=r"[^,]+", max_features=6000)
    P = vec_people.fit_transform(
        (items_df["cast_csv"].fillna("") + "," + items_df["directors_csv"].fillna("")).str.strip(",")
    )

    # title + summary
    vec_text = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)
    T = vec_text.fit_transform(items_df["text"].fillna(""))

    # year buckets
    items_df["year_bucket"] = _year_bucket_feature(items_df["year"])
    vec_year = CountVectorizer(token_pattern=r"[^,]+")
    Y = vec_year.fit_transform(items_df["year_bucket"].fillna("year_unknown"))

    # weighted stack
    mats, weights = [], []
    if C.shape[1] > 0 and FEAT_W_COLLECTIONS > 0:
        mats.append(C)
        weights.append(FEAT_W_COLLECTIONS)
    if G.shape[1] > 0 and FEAT_W_GENRES > 0:
        mats.append(G)
        weights.append(FEAT_W_GENRES)
    if P.shape[1] > 0 and FEAT_W_PEOPLE > 0:
        mats.append(P)
        weights.append(FEAT_W_PEOPLE)
    if T.shape[1] > 0 and FEAT_W_TEXT > 0:
        mats.append(T)
        weights.append(FEAT_W_TEXT)
    if Y.shape[1] > 0 and FEAT_W_YEAR > 0:
        mats.append(Y)
        weights.append(FEAT_W_YEAR)

    if not mats:
        raise RuntimeError("No feature matrices produced; check inputs/weights")

    mats = [m.multiply(w) for m, w in zip(mats, weights)]
    X = sp.hstack(mats, format="csr")
    X = normalize(X)

    dump({
        "vec_collections": vec_collections,
        "vec_genres": vec_genres,
        "vec_people": vec_people,
        "vec_text": vec_text,
        "vec_year": vec_year,
        "weights": {
            "collections": FEAT_W_COLLECTIONS,
            "genres": FEAT_W_GENRES,
            "people": FEAT_W_PEOPLE,
            "text": FEAT_W_TEXT,
            "year": FEAT_W_YEAR,
            "year_bucket_size": YEAR_BUCKET_SIZE,
        },
    }, VEC_PATH)

    sp.save_npz(MAT_PATH, X)
    items_df[["item_id"]].astype(str).to_csv(IDX_PATH, index=False)
    return X

def load_artifacts():
    vecs = load(VEC_PATH)
    X = sp.load_npz(MAT_PATH)
    id_index = pd.read_csv(IDX_PATH, dtype={"item_id": str})["item_id"].astype(str).tolist()
    return vecs, X, id_index