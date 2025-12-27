# base/agents/scheduler.py
from __future__ import annotations

import asyncio
import threading
import time

from apscheduler.schedulers.background import BackgroundScheduler
from collections.abc import Callable
from loguru import logger
from typing import TYPE_CHECKING, Any

from base.learning.habit_miner import HabitMiner
from base.memory.store import MemoryStore


if TYPE_CHECKING:
    from base.learning.habit_miner import HabitMiner
    from base.memory.store import MemoryStore
    

class Scheduler:
    """
    Two scheduling mechanisms live here today:

    1) APScheduler (cron-style jobs) via self.scheduler
    2) A simple interval task loop (threaded) via self.tasks + self._loop()

    This refactor makes the interval-loop deterministic and unit-testable by allowing:
      - injected clock/sleeper
      - injected miner (so tests don't import HabitMiner/spaCy)
      - injected APScheduler instance (so tests can stub/spy)
    """

    # def __init__(self):
    #   self.db = SQLiteConn(settings.db_path)
    #   self.scheduler = BackgroundScheduler()
    #   self.miner = HabitMiner(self.db)
    # def __init__(self, db, memory, store: MemoryStore):
    #     self.db = db
    #     self.store = store
    #     self.memory = memory
    #     self.miner = HabitMiner(memory=memory, store=store, db=db)
    #     self.scheduler = BackgroundScheduler()
    #     self.tasks: dict[str, dict] = {}
    #     self.running = False
    def __init__(
        self,
        db: Any,
        memory: Any,
        store: "MemoryStore",
        *,
        miner: "HabitMiner | None" = None,
        apscheduler: BackgroundScheduler | None = None,
        clock: Callable[[], float] = time.time,
        sleeper: Callable[[float], None] = time.sleep,
        loop_sleep_sec: float = 1.0,
    ):
        self.db = db
        self.store = store
        self.memory = memory

        # Lazy-import HabitMiner only if we actually need it
        if miner is None:
            from base.learning.habit_miner import HabitMiner  # local import to keep tests light

            miner = HabitMiner(memory=memory, store=store, db=db)

        self.miner = miner
        self.scheduler = apscheduler or BackgroundScheduler()

        self._clock = clock
        self._sleep = sleeper
        self._loop_sleep_sec = loop_sleep_sec

        self.tasks: dict[str, dict[str, Any]] = {}
        self.running = False
        self._thread: threading.Thread | None = None
        

    async def run_periodic(self, interval_sec: int = 86400):
        """Legacy async loop (kept)."""
        while True:
            logger.info("Scheduler: running HabitMiner...")
            self.miner.mine()
            await asyncio.sleep(interval_sec)

    def add_daily(self, func: Callable[[], None], hour: int = 3, minute: int = 0, *, job_id: str | None = None):
        """Run `func` once a day at given hour/minute. Idempotent via job_id."""
        jid = job_id or f"daily:{getattr(func, '__name__', 'job')}"
        self.scheduler.add_job(
            func,
            "cron",
            hour=hour,
            minute=minute,
            id=jid,
            replace_existing=True,
        )
        logger.info(f"Scheduled daily job {jid} at {hour:02d}:{minute:02d}")

    def add_task(self, name: str, interval: int, func: Callable[[], None]) -> None:
        """Register/replace an interval task for the threaded loop."""
        self.tasks[name] = {
            "interval": interval,
            "func": func,
            "last_run": self._clock(),
        }
        logger.info(f"[Scheduler] Registered task: {name} every {interval}s")

    def run_pending(self, *, now: float | None = None) -> None:
        """
        Run any due interval tasks once.
        Public to make deterministic unit tests possible.
        """
        current = self._clock() if now is None else now

        for name, task in list(self.tasks.items()):
            last_run = float(task["last_run"])
            interval = int(task["interval"])

            if current - last_run >= interval:
                try:
                    task["func"]()
                except Exception as e:
                    logger.exception(f"[Scheduler] Task {name} failed: {e}")
                finally:
                    task["last_run"] = current
                    
    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # def _loop(self):
    #     while self.running:
    #         now = time.time()
    #         for name, task in self.tasks.items():
    #             if now - task["last_run"] >= task["interval"]:
    #                 try:
    #                     task["func"]()
    #                 except Exception as e:
    #                     print(f"[Scheduler] Task {name} failed: {e}")
    #                 task["last_run"] = now
    #         time.sleep(1)

    def _loop(self) -> None:
        while self.running:
            self.run_pending()
            self._sleep(self._loop_sleep_sec)

    def stop(self) -> None:
        self.running = False

        # Do not hang on shutdown; thread is daemon anyway
        try:
            self.scheduler.shutdown(wait=False)
        except Exception as e:
            logger.warning(f"[Scheduler] shutdown() raised: {e}")

        logger.info("Scheduler stopped.")
