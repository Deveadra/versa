# tests/scheduler/test_scheduler_run_periodic.py

from __future__ import annotations

import asyncio
import pytest

from base.agents.scheduler import Scheduler


class DummyMiner:
    def __init__(self):
        self.called = 0

    def mine(self):
        self.called += 1


class DummyAps:
    def shutdown(self, wait=False):
        return None


@pytest.mark.asyncio
async def test_run_periodic_calls_mine_once_then_cancel(monkeypatch):
    miner = DummyMiner()

    s = Scheduler(
        db=None,
        memory=None,
        store=None,
        miner=miner,
        apscheduler=DummyAps(),
        clock=lambda: 0.0,
        sleeper=lambda _sec: None,
    )

    async def fake_sleep(_sec):
        raise asyncio.CancelledError()

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        await s.run_periodic(interval_sec=999)

    assert miner.called == 1
