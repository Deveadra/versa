
from __future__ import annotations
import json
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Iterable
from dateutil.rrule import rrulestr, rrule, WEEKLY, DAILY, MONTHLY
from dateutil import tz
from ..database.sqlite import SQLiteConn

class CalendarStore:
    def __init__(self, db: SQLiteConn):
        self.db = db
        self._init_tables()

    def _init_tables(self):
        self.db.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS calendar_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                start TEXT NOT NULL,          -- ISO UTC
                end TEXT,                     -- ISO UTC
                rrule TEXT,                   -- RFC5545 RRULE (optional)
                location TEXT,
                attendees TEXT,               -- JSON array of strings or name/email pairs
                meta TEXT                     -- JSON blob for anything else
            );
            """
        )
        self.db.conn.commit()

    # -----------------------------
    # CRUD
    # -----------------------------
    def add_event(
        self,
        title: str,
        start_iso: str,
        end_iso: Optional[str] = None,
        rrule_str: Optional[str] = None,
        location: Optional[str] = None,
        attendees: Optional[Iterable[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> int:
        cur = self.db.conn.execute(
            """
            INSERT INTO calendar_events (title, start, end, rrule, location, attendees, meta)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                title,
                start_iso,
                end_iso,
                rrule_str,
                location,
                json.dumps(list(attendees) if attendees else []),
                json.dumps(meta or {}),
            ),
        )
        self.db.conn.commit()
        if cur.lastrowid is None:
            raise RuntimeError("Failed to insert event: lastrowid is None")
        return cur.lastrowid

    def list_events(self) -> List[dict]:
        cur = self.db.conn.execute("SELECT id, title, start, end, rrule, location, attendees, meta FROM calendar_events")
        rows = []
        for r in cur.fetchall():
            rows.append({
                "id": r["id"], "title": r["title"], "start": r["start"], "end": r["end"],
                "rrule": r["rrule"], "location": r["location"],
                "attendees": json.loads(r["attendees"] or "[]"),
                "meta": json.loads(r["meta"] or "{}"),
            })
        return rows

    # -----------------------------
    # Expansion
    # -----------------------------
    def expand(self, window_start_iso: str, window_end_iso: str, limit: int = 200) -> List[dict]:
        """
        Expand one-off and recurring events into concrete instances within [start, end].
        """
        wstart = datetime.fromisoformat(window_start_iso.replace("Z", "+00:00")).astimezone(tz.UTC)
        wend = datetime.fromisoformat(window_end_iso.replace("Z", "+00:00")).astimezone(tz.UTC)

        out: List[dict] = []
        for e in self.list_events():
            base_start = datetime.fromisoformat(e["start"].replace("Z", "+00:00")).astimezone(tz.UTC)
            base_end = datetime.fromisoformat(e["end"].replace("Z", "+00:00")).astimezone(tz.UTC) if e["end"] else None

            if not e["rrule"]:
                # one-off: include if overlaps window
                if self._overlaps(base_start, base_end, wstart, wend):
                    out.append({
                        "title": e["title"],
                        "start": base_start.isoformat(),
                        "end": (base_end or (base_start + timedelta(hours=1))).isoformat(),
                        "location": e["location"],
                        "attendees": e["attendees"],
                        "meta": e["meta"],
                        "event_id": e["id"],
                        "recurrence": None,
                    })
                continue

            # recurring: expand occurrences
            rule = rrulestr(e["rrule"], dtstart=base_start)
            count = 0
            for dt in rule.between(wstart - timedelta(days=1), wend + timedelta(days=1), inc=True):
                # create instance per occurrence within window
                inst_start = dt.astimezone(tz.UTC)
                inst_end = (inst_start + ( (base_end - base_start) if base_end else timedelta(hours=1) )).astimezone(tz.UTC)
                if self._overlaps(inst_start, inst_end, wstart, wend):
                    out.append({
                        "title": e["title"],
                        "start": inst_start.isoformat(),
                        "end": inst_end.isoformat(),
                        "location": e["location"],
                        "attendees": e["attendees"],
                        "meta": e["meta"],
                        "event_id": e["id"],
                        "recurrence": e["rrule"],
                    })
                    count += 1
                    if count >= limit:
                        break
        # sort by start time
        out.sort(key=lambda x: x["start"])
        return out

    @staticmethod
    def _overlaps(a_start: datetime, a_end: Optional[datetime], b_start: datetime, b_end: datetime) -> bool:
        a_end = a_end or (a_start + timedelta(hours=1))
        return not (a_end <= b_start or a_start >= b_end)
