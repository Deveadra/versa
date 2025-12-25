# tests/learning/test_habit_miner.py


from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from base.learning.habit_miner import HabitMiner


@pytest.fixture
def dummy_store():
    store = MagicMock()
    store.conn.execute.return_value.fetchall.return_value = [
        {"content": "play jazz", "ts": (datetime.utcnow() - timedelta(days=1)).isoformat()},
        {"content": "play jazz", "ts": (datetime.utcnow() - timedelta(days=1)).isoformat()},
        {"content": "turn on light", "ts": (datetime.utcnow() - timedelta(days=1)).isoformat()},
    ]
    return store


@pytest.fixture
def dummy_db():
    db = MagicMock()
    db.conn.execute.return_value.fetchall.return_value = [
        ["play jazz", (datetime.utcnow() - timedelta(days=1)).isoformat()],
        ["play jazz", (datetime.utcnow() - timedelta(days=2)).isoformat()],
        ["play jazz", (datetime.utcnow() - timedelta(days=3)).isoformat()],
        ["turn on light", (datetime.utcnow() - timedelta(days=4)).isoformat()],
    ]
    return db


@pytest.fixture
def habit_miner(dummy_db, dummy_store):
    memory = MagicMock()
    miner = HabitMiner(db=dummy_db, memory=memory, store=dummy_store)
    miner.save_profile = MagicMock()
    miner.load_profile = MagicMock(return_value={})
    return miner


def test_learn_and_predict(habit_miner):
    habit_miner.learn("play jazz")
    next_time = habit_miner.predict_next("play jazz")
    assert isinstance(next_time, datetime)


def test_summarize_and_get_summaries(habit_miner):
    candidates = habit_miner.extract_candidates()
    summaries = habit_miner.summarize(candidates)
    assert any("User often says" in s for s in summaries)


def test_check_upcoming(habit_miner):
    habit_miner.habits = [
        {"action": "play jazz", "time": (datetime.utcnow() + timedelta(minutes=10)).time()}
    ]
    result = habit_miner.check_upcoming()
    assert isinstance(result, list)
    assert len(result) >= 1


def test_reinforce_and_adjust(habit_miner):
    habit_miner.reinforce("play jazz")
    habit_miner.adjust("turn on light")
    assert habit_miner.save_profile.called


def test_export_summary(habit_miner):
    habit_miner.load_profile.return_value = {
        "most_used_commands": ["music", "lights"],
        "tone_bias": "succinct",
        "persona_summary": "Reinforced preference: play jazz (x3).",
    }
    summary = habit_miner.export_summary()
    assert "music" in summary

    succinct_markers = ("succinct", "concise", "short", "brief", "long answers")
    assert any(m in summary.lower() for m in succinct_markers)


def test_export_summary_missing_tone_bias(habit_miner):
    habit_miner.load_profile.return_value = {
        "most_used_commands": ["music", "lights"],
        "persona_summary": "Reinforced preference: play jazz (x3).",
    }
    summary = habit_miner.export_summary()
    assert isinstance(summary, str)
    assert "music" in summary

    summary = habit_miner.export_summary()
    assert "music" in summary
    assert "Dislikes long answers" in summary


def test_prune_habits(habit_miner):
    habit_miner.load_profile.return_value = {
        "reinforcements": {"play jazz": 4},
        "adjustments": {"turn on light": 4},
    }
    habit_miner.prune_habits()
    assert habit_miner.save_profile.called
