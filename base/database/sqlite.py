
from __future__ import annotations
import sqlite3
from pathlib import Path
from loguru import logger
# from base.database import migrations

class SQLiteConn:
    def __init__(self, path: str):
        self.path = Path(path)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        logger.info("Running migrations if needed")
        mig_dir = Path(__file__).parent / "migrations"
        files = sorted(p for p in mig_dir.glob("*.sql"))
        sql = (Path(__file__).parent / "migrations" / "0001_init.sql").read_text(encoding="utf-8")

        for p in files:
            sql = p.read_text(encoding="utf-8")
            self.conn.executescript(sql)
        self.conn.executescript(sql)
        self.conn.commit()

    def cursor(self):
        return self.conn.cursor()

    def close(self):
        self.conn.close()