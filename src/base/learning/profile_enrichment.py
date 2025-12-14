from __future__ import annotations

from typing import Any

from loguru import logger

from base.core.profile_manager import ProfileManager
from base.learning.habit_miner import HabitMiner


class ProfileEnricher:
    def __init__(self, pm: ProfileManager, miner: HabitMiner):
        self.pm = pm
        self.miner = miner

    def run(self) -> dict[str, Any]:
        profile = self.pm.load_profile()
        top_svc = self.miner.top("music.service=", 1)
        if top_svc:
            svc = top_svc[0]["key"].split("=", 1)[1]
            profile.setdefault("preferred_player", svc)
            profile["preferred_player"] = svc
        top_genre = self.miner.top("music.genre=", 1)
        if top_genre:
            genre = top_genre[0]["key"].split("=", 1)[1]
            profile.setdefault("favorite_music", genre)
            profile["favorite_music"] = genre
        top_greet = self.miner.top("ux.greeting_style=", 1)
        if top_greet:
            gs = top_greet[0]["key"].split("=", 1)[1]
            profile.setdefault("greeting_style", gs)
            profile["greeting_style"] = gs
        top_sleep = self.miner.top("user.sleep_time=", 1)
        if top_sleep:
            st = top_sleep[0]["key"].split("=", 1)[1]
            profile.setdefault("sleep_time", st)
            profile["sleep_time"] = st
        self.pm.save_profile(profile)
        logger.info(
            "ProfileEnricher wrote fields: preferred_player, favorite_music, greeting_style, sleep_time"
        )
        return profile
