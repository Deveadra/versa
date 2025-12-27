# tests/scheduler/test_scheduler_interval_tasks.py

from __future__ import annotations

from base.agents.scheduler import Scheduler


class DummyMiner:
    def mine(self):
        return None


class DummyAps:
    def add_job(self, *args, **kwargs):
        raise AssertionError("APS should not be touched in these tests")

    def shutdown(self, wait=False):
        return None


def test_run_pending_runs_only_when_due():
    calls: list[str] = []

    def job():
        calls.append("ran")

    s = Scheduler(
        db=None,
        memory=None,
        store=None,  # not used by these tests
        miner=DummyMiner(),
        apscheduler=DummyAps(),
        clock=lambda: 0.0,
        sleeper=lambda _sec: None,
    )

    s.add_task("t1", interval=10, func=job)

    s.run_pending(now=9.9)
    assert calls == []

    s.run_pending(now=10.0)
    assert calls == ["ran"]

    # calling again without time moving should not rerun
    s.run_pending(now=10.0)
    assert calls == ["ran"]


def test_run_pending_continues_after_exception():
    calls: list[str] = []

    def bad():
        raise RuntimeError("boom")

    def good():
        calls.append("good")

    s = Scheduler(
        db=None,
        memory=None,
        store=None,
        miner=DummyMiner(),
        apscheduler=DummyAps(),
        clock=lambda: 0.0,
        sleeper=lambda _sec: None,
    )

    s.add_task("bad", interval=1, func=bad)
    s.add_task("good", interval=1, func=good)

    # Should not raise, and should still run the "good" task
    s.run_pending(now=1.1)
    assert calls == ["good"]
