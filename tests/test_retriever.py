import os
import sqlite3
import tempfile
import unittest

from base.llm.retriever import DbRetriever


class RetrieverTests(unittest.TestCase):
    def setUp(self):
        self.dbfile = tempfile.NamedTemporaryFile(delete=False).name
        self.conn = sqlite3.connect(self.dbfile, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        cur = self.conn.cursor()
        cur.executescript(
            """
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT, value TEXT, created_at TEXT, last_reinforced TEXT
        );
        CREATE TABLE IF NOT EXISTS usage_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_text TEXT, normalized_intent TEXT, resolved_action TEXT, params_json TEXT, created_at TEXT
        );
        """
        )
        self.conn.commit()
        # insert sample data
        cur.execute(
            "INSERT INTO facts (key, value, created_at) VALUES (?, ?, datetime('now', '-2 days'))",
            ("music_pref", "spotify"),
        )
        cur.execute(
            "INSERT INTO facts (key, value, created_at) VALUES (?, ?, datetime('now', '-20 days'))",
            ("sleep_time", "23:30"),
        )
        cur.execute(
            "INSERT INTO usage_log (user_text, normalized_intent, resolved_action, params_json, created_at) VALUES (?, ?, ?, ?, datetime('now', '-1 days'))",
            (
                "Play lo-fi on Spotify",
                "music.play",
                "play:music",
                '{"service":"spotify","genre":"lo-fi"}',
            ),
        )
        self.conn.commit()

    def tearDown(self):
        try:
            os.unlink(self.dbfile)
        except Exception:
            pass

    def test_query_matches(self):
        class ConnWrapper:
            def __init__(self, conn):
                self.conn = conn

            def cursor(self):
                return self.conn.cursor()

        r = DbRetriever(ConnWrapper(self.conn))
        results = r.query("play lo-fi", top_k=3)
        summaries = [r["summary"] for r in results]
        self.assertTrue(any("usage:" in s for s in summaries))
        self.assertTrue(any("fact:" in s or "music_pref" in s for s in summaries))


if __name__ == "__main__":
    unittest.main()
