# Recommenderr

Content-based movie & series recommendations from your **Plex** watch history.
Runs as a single Docker container, stores artifacts on a bind-mounted `/data`, and exposes a simple CLI for ingestion, model building, and fetching recommendations (with optional explanations).

---

## ✨ Features

* **Direct Plex ingest**: pulls watch history from your Plex Media Server and catalogs your Movie & TV libraries.
* **User mapping**: resolves Plex Home users from `plex.tv` to local watch events.
* **Series-level history**: TV episodes are canonicalized to the **series** id, so recs are at the show/movie level.
* **Content features**: genres, cast, directors, **collections**, title/summary TF-IDF, and **year buckets**.
* **Weighted model**: feature groups have tunable weights via environment variables.
* **Cosine similarity** ranking + **percentile “match” score** (0–100).
* **Explain mode**: per-feature group contribution breakdowns with `--explain`.
* **Scheduler**: auto-ingest and rebuild on an interval; disable for fast dev loops.
* **Durable artifacts**: vectors and indices saved to `/data` (shared with the DB).

---

## 🧠 How it works

1. **Ingest**

   * `GET /status/sessions/history/all` (PMS): watch events
     Episodes → mapped to their **series** id (grandparent).
   * `GET /library/sections/*`: enumerates all movies & shows (candidate pool).
   * `GET /library/metadata/{id}`: metadata used for text/features.
   * `GET https://plex.tv/api/home/users`: maps account id → username/display name.

2. **Preferences**
   For each `(user_id, item_id)` we compute a preference:

   * amount watched (log events) • recency decay (~90d) • duration proxy • small bias

3. **Vectorization**
   Items are embedded as a weighted concatenation of:

   * **Collections** (CountVectorizer)
   * **Genres** (CountVectorizer)
   * **People**: cast + directors (CountVectorizer)
   * **Title + Summary** (TF-IDF, 1–2 grams)
   * **Year buckets** (CountVectorizer over bucket labels)

   Matrices are L2-normalized; cosine similarity is then just a dot product.

4. **User vector**
   A user is the recency-weighted average of their recent liked items’ vectors.

5. **Ranking**
   Unwatched candidates are scored by cosine to the user vector. We return:

   * `score`: raw cosine (0..1-ish)
   * `relevancy`: percentile (0–100), where the very top result shows **100%**

---

## 📦 Requirements

* Plex Media Server reachable from the container
* A **Plex token** with permission to read libraries & history
* Docker (or Python 3.11 if running natively)

---

## ⚙️ Configuration

Create a `.env` next to your Compose file or inject as container env.
See `.env.example` for defaults.

| Variable            | Default                    | Notes                                       |
| ------------------- | -------------------------- | ------------------------------------------- |
| `PLEX_BASE`         | `http://localhost:32400`   | Your PMS URL                                |
| `PLEX_TOKEN`        | *(required)*               | Plex auth token                             |
| `DB_PATH`           | `/data/recommendations.db` | SQLite path                                 |
| `ART_DIR`           | `/data`                    | Where vector artifacts are stored           |
| `PULL_INTERVAL_MIN` | `60`                       | Scheduler interval (minutes)                |
| `DISABLE_AUTOSTART` | `false`                    | Skip auto-pipeline on start (useful in dev) |
| `LOG_LEVEL`         | `INFO`                     | `DEBUG` for more detail                     |
| `LOG_FILE`          | `/data/recommenderr.log`   | Rotating log file                           |

**Feature weights:**

```
FEAT_W_GENRES=1.0
FEAT_W_PEOPLE=0.8
FEAT_W_TEXT=0.6
FEAT_W_COLLECTIONS=1.4
FEAT_W_YEAR=0.5
YEAR_BUCKET_SIZE=5
YEAR_MIN=1900
YEAR_MAX=2030
```

---

##
