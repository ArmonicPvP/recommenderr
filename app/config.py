import os
from dotenv import load_dotenv
load_dotenv()

PLEX_BASE = os.getenv("PLEX_BASE", "http://localhost:32400").rstrip("/")
PLEX_TOKEN = os.getenv("PLEX_TOKEN", "")

DB_PATH = os.getenv("DB_PATH", "/data/recommendations.db")
ART_DIR = os.getenv("ART_DIR", "/data")
PULL_INTERVAL_MIN = int(os.getenv("PULL_INTERVAL_MIN", "60"))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "/data/recommenderr.log")

DISABLE_AUTOSTART = os.getenv("DISABLE_AUTOSTART", "false").lower() in ("1", "true", "yes")

# feature weights (can be tuned via env)
FEAT_W_GENRES      = float(os.getenv("FEAT_W_GENRES", "0.5"))
FEAT_W_PEOPLE      = float(os.getenv("FEAT_W_PEOPLE", "1.0"))
FEAT_W_TEXT        = float(os.getenv("FEAT_W_TEXT", "0.8"))
FEAT_W_COLLECTIONS = float(os.getenv("FEAT_W_COLLECTIONS", "2.0"))
FEAT_W_YEAR        = float(os.getenv("FEAT_W_YEAR", "0.4"))

REC_COLLECTION_BOOST     = float(os.getenv("REC_COLLECTION_BOOST", "0.25"))
REC_COLLECTION_LOOKBACK  = int(os.getenv("REC_COLLECTION_LOOKBACK", "50"))

# year bucketing
YEAR_BUCKET_SIZE   = int(os.getenv("YEAR_BUCKET_SIZE", "5"))
YEAR_MIN           = int(os.getenv("YEAR_MIN", "1900"))
YEAR_MAX           = int(os.getenv("YEAR_MAX", "2030"))

assert PLEX_TOKEN, "Set PLEX_TOKEN"