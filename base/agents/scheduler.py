
from __future__ import annotations
from apscheduler.schedulers.background import BackgroundScheduler
from loguru import logger


class UltronScheduler:
  def __init__(self):
    self.scheduler = BackgroundScheduler()


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