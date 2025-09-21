# conn_wrap.py
import sqlite3
from typing import Final

class ConnWrap:
    def __init__(self, conn: sqlite3.Connection):
        self._conn: Final[sqlite3.Connection] = conn

    @property
    def sqlite(self) -> sqlite3.Connection:
        return self._conn
