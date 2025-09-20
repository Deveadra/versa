# Simple simulator that logs events to a local sqlite (no repo imports required)
import sqlite3
import os
import random
import json

DB = '/tmp/ultron_local_sim.db'

try:
  os.remove(DB)
except Exception:
  pass

conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

cur.executescript("""
CREATE TABLE IF NOT EXISTS usage_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_text TEXT,
  normalized_intent TEXT,
  resolved_action TEXT,
  params_json TEXT,
  success INTEGER,
  latency_ms INTEGER,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS habits (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  key TEXT NOT NULL,
  count INTEGER NOT NULL DEFAULT 0,
  score REAL NOT NULL DEFAULT 0.0,
  last_used DATETIME,
  UNIQUE(key)
);
CREATE TABLE IF NOT EXISTS facts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  key TEXT,
  value TEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
""")
conn.commit()

for i in range(50):
  svc = random.choices(['spotify', 'youtube_music'], weights=[0.8, 0.2])[0]
  genre = random.choices(['lo-fi', 'jazz', 'vaporwave'], weights=[0.7, 0.2, 0.1])[0]
  params = json.dumps({'service': svc, 'genre': genre})
  cur.execute(
    "INSERT INTO usage_log (user_text, normalized_intent, resolved_action, params_json, success, latency_ms) VALUES (?, ?, ?, ?, ?, ?)",
    (f'Play {genre}', 'music.play', 'play:music', params, 1, random.randint(30, 300))
  )

conn.commit()
print('Wrote 50 simulated usage events to', DB)
