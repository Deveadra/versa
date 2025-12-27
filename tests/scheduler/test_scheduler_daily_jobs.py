# tests/scheduler/test_scheduler_daily_jobs.py

from __future__ import annotations

from base.agents.scheduler import Scheduler


class DummyMiner:
    def mine(self):
        return None


class SpyAps:
    def __init__(self):
        self.added = []
        self.shutdown_called = False

    def add_job(self, func, trigger, **kwargs):
        self.added.append((func, trigger, kwargs))

    def shutdown(self, wait=False):
        self.shutdown_called = True


def test_add_daily_is_idempotent_via_replace_existing():
    aps = SpyAps()

    s = Scheduler(
        db=None,
        memory=None,
        store=None,
        miner=DummyMiner(),
        apscheduler=aps,
        clock=lambda: 0.0,
        sleeper=lambda _sec: None,
    )

    def hello():
        return None

    s.add_daily(hello, hour=3, minute=5, job_id="daily:hello")
    s.add_daily(hello, hour=3, minute=5, job_id="daily:hello")

    assert len(aps.added) == 2
    # second call should be replace_existing=True as well
    assert aps.added[0][1] == "cron"
    assert aps.added[0][2]["hour"] == 3
    assert aps.added[0][2]["minute"] == 5
    assert aps.added[0][2]["id"] == "daily:hello"
    assert aps.added[0][2]["replace_existing"] is True
