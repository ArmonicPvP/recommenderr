CREATE TABLE IF NOT EXISTS items(
  item_id TEXT PRIMARY KEY,
  type TEXT, title TEXT, year INTEGER, runtime INTEGER,
  summary TEXT, genres_csv TEXT, cast_csv TEXT,
  directors_csv TEXT, collections_csv TEXT, poster_url TEXT
);

CREATE TABLE IF NOT EXISTS watch_events(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id TEXT, user_name TEXT,
  item_id TEXT,
  started_at TEXT, stopped_at TEXT,
  duration INTEGER, view_offset INTEGER,
  completed INTEGER,
  source TEXT,
  plex_history_key TEXT
);
CREATE INDEX IF NOT EXISTS idx_we_user_time ON watch_events(user_id, started_at);
CREATE UNIQUE INDEX IF NOT EXISTS uq_we_source_history
ON watch_events(source, plex_history_key)
WHERE plex_history_key IS NOT NULL AND plex_history_key != '';
CREATE UNIQUE INDEX IF NOT EXISTS uq_we_fallback
ON watch_events(user_id, item_id, started_at, source);

CREATE TABLE IF NOT EXISTS user_item_pref(
  user_id TEXT, item_id TEXT, preference REAL, last_seen_at TEXT,
  PRIMARY KEY (user_id, item_id)
);

CREATE TABLE IF NOT EXISTS users(
  user_id TEXT PRIMARY KEY,
  user_name TEXT,
  display_name TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_name ON users(user_name);
