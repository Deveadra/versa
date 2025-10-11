# base/agents/scheduler.py
from __future__ import annotations

import asyncio
import threading
import time

from apscheduler.schedulers.background import BackgroundScheduler
from collections.abc import Callable
from loguru import logger

from base.agents.dream import DreamCycle
from base.learning.habit_miner import HabitMiner
from base.memory.store import MemoryStore
from base.self_improve.diagnostic_engine import DiagnosticEngine
from base.self_improve.proposal_engine import ProposalEngine


class Scheduler:
    # def __init__(self):
    #   self.db = SQLiteConn(settings.db_path)
    #   self.scheduler = BackgroundScheduler()
    #   self.miner = HabitMiner(self.db)
    def __init__(self, db, memory, store: MemoryStore):
        self.db = db
        self.store = store
        self.memory = memory
        self.miner = HabitMiner(memory=memory, store=store, db=db)
        self.scheduler = BackgroundScheduler()
        self.tasks: dict[str, dict] = {}
        self.running = False

    async def run_periodic(self, interval_sec: int = 86400):  # default: once/day
        while True:
            logger.info("Scheduler: running HabitMiner...")
            self.miner.mine()
            await asyncio.sleep(interval_sec)

    def add_daily(self, func, hour: int = 3, minute: int = 0):
        """Run `func` once a day at given hour/minute."""
        self.scheduler.add_job(func, "cron", hour=hour, minute=minute)
        logger.info(f"Scheduled daily job {func.__name__} at {hour:02d}:{minute:02d}")

    def add_task(self, name: str, interval: int, func: Callable[[], None]):
        """Add or update a repeating background task."""
        self.tasks[name] = {"interval": interval, "func": func, "last_run": 0}

    def start(self):
        if self.running:
            return
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            now = time.time()
            for name, task in self.tasks.items():
                if now - task["last_run"] >= task["interval"]:
                    try:
                        task["func"]()
                    except Exception as e:
                        print(f"[Scheduler] Task {name} failed: {e}")
                    task["last_run"] = now
            time.sleep(1)

    def stop(self):
        self.scheduler.shutdown()
        logger.info("Scheduler stopped")
