from __future__ import annotations
from typing import Dict, Any, Optional, List
from loguru import logger
from base.core.profile_manager import ProfileManager
from base.learning.habit_miner import HabitMiner
from base.database.sqlite import SQLiteConn
import json

class PersonaPrimer:
    def __init__(self, profile_mgr: ProfileManager, miner: HabitMiner, db: SQLiteConn):
        self.pm = profile_mgr
        self.miner = miner
        self.db = db

    def build(self, user_text: str) -> str:
        """Return a short persona primer string (3-6 lines) describing user prefs and recent habits."""
        try:
            profile = self.pm.load_profile() if self.pm else {}
        except Exception:
            profile = {}
        pieces: List[str] = []
        # basic profile fields
        if profile.get('name'):
            pieces.append(f"Name: {profile.get('name')}")
        if profile.get('preferred_player'):
            pieces.append(f"preferred_player: {profile.get('preferred_player')}")
        if profile.get('favorite_music'):
            pieces.append(f"favorite_music: {profile.get('favorite_music')}")
        if profile.get('greeting_style'):
            pieces.append(f"greeting_style: {profile.get('greeting_style')}")
        if profile.get('sleep_time'):
            pieces.append(f"sleep_time: {profile.get('sleep_time')}")
        # top habits
        try:
            if self.miner:
                top_music = self.miner.top('music.service=', 1)
                if top_music:
                    pieces.append('habit: ' + top_music[0]['key'])
                top_genre = self.miner.top('music.genre=', 1)
                if top_genre:
                    pieces.append('habit: ' + top_genre[0]['key'])
                top_greet = self.miner.top('ux.greeting_style=', 1)
                if top_greet:
                    pieces.append('habit: ' + top_greet[0]['key'])
        except Exception:
            pass
        # lightweight facts: pick top 3 most recently reinforced facts
        try:
            c = self.db.conn.cursor()
            c.execute("SELECT key, value FROM facts ORDER BY COALESCE(last_reinforced, created_at) DESC LIMIT 3")
            rows = c.fetchall()
            for r in rows:
                pieces.append(f"fact: {r['key']}={r['value']}")
        except Exception:
            pass
        # limit to reasonable length
        if not pieces:
            return ''
        # join into a short block
        return '\n'.join(pieces[:6])
