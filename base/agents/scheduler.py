
from __future__ import annotations
import asyncio

from apscheduler.schedulers.background import BackgroundScheduler
from base.learning.habit_miner import HabitMiner
from loguru import logger
from database.sqlite import SQLiteConn
from config.config import settings


class Scheduler:
  # def __init__(self):
  #   self.db = SQLiteConn(settings.db_path)
  #   self.scheduler = BackgroundScheduler()
  #   self.miner = HabitMiner(self.db)
  def __init__(self, db):
      self.db = db
      self.miner = HabitMiner(db)
      self.scheduler = BackgroundScheduler()


  async def run_periodic(self, interval_sec: int = 86400):  # default: once/day
          while True:
              logger.info("Scheduler: running HabitMiner...")
              self.miner.mine()
              await asyncio.sleep(interval_sec)
              
  def add_daily(self, func, hour: int = 3, minute: int = 0):
    """Run `func` once a day at given hour/minute."""
    self.scheduler.add_job(func, 'cron', hour=hour, minute=minute)
    logger.info(f"Scheduled daily job {func.__name__} at {hour:02d}:{minute:02d}")


  def start(self):
    self.scheduler.start()
    logger.info("Scheduler started")


  def stop(self):
    self.scheduler.shutdown()
    logger.info("Scheduler stopped")