# tests/scheduler/test_scheduler_start_stop.py

from __future__ import annotations

import base.agents.scheduler as scheduler_mod


class DummyMiner:
    def mine(self):
        return None


class SpyAps:
    def __init__(self):
        self.shutdown_called = False

    def shutdown(self, wait=False):
        self.shutdown_called = True


def test_start_is_idempotent(monkeypatch):
    started = []

    class FakeThread:
        def __init__(self, target, daemon):
            self.target = target
            self.daemon = daemon

        def start(self):
            started.append(self)

    monkeypatch.setattr(scheduler_mod.threading, "Thread", FakeThread)

    s = scheduler_mod.Scheduler(
        db=None,
        memory=None,
        store=None,
        miner=DummyMiner(),
        apscheduler=SpyAps(),
        clock=lambda: 0.0,
        sleeper=lambda _sec: None,
    )

    s.start()
    s.start()
    assert len(started) == 1


def test_stop_calls_shutdown_and_sets_running_false():
    aps = SpyAps()

    s = scheduler_mod.Scheduler(
        db=None,
        memory=None,
        store=None,
        miner=DummyMiner(),
        apscheduler=aps,
        clock=lambda: 0.0,
        sleeper=lambda _sec: None,
    )

    s.running = True
    s.stop()
    assert s.running is False
    assert aps.shutdown_called is True
