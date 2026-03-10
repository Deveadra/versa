# main.py

from __future__ import annotations

import atexit
import os
import random
import shutil
import signal
import sqlite3
import threading
import time
from contextlib import suppress
from datetime import UTC, datetime
from http import HTTPStatus

import requests

from base.agents.scheduler import Scheduler
from base.calendar import calendar_flow
from base.core.audio import interrupt, listen_for_wake_word, listen_until_silence, stream_speak
from base.core.commands import handle_policy_command
from base.core.core import SLEEP_WORDS, STOP_WORDS, AerithState, reset_session
from base.core.decider import Decider
from base.core.mode_classifier import classify_mode
from base.core.plugin_manager import PluginManager
from base.core.profile import get_pref, get_profile
from base.core.profile_manager import ProfileManager
from base.database.sqlite import SQLiteConn
from base.learning.engagement_manager import EngagementManager
from base.learning.habit_miner import HabitMiner
from base.llm.brain import ask_aerith_stream
from base.memory.recall import format_memories, recall_relevant
from base.memory.store import MemoryStore
from base.plugins import email_flow_original, file_manager, media_smart_home, system
from base.policy.consequence_linker import link_consequence
from base.policy.context_signals import ContextSignals
from base.policy.feedback import record_feedback, schedule_signal_check
from base.policy.policy_store import PolicyStore
from base.repl import commands as repl_commands
from base.voice.tts_elevenlabs import Voice
from config.config import settings
from personalities.loader import load_personality

# ------------------------------
# Constants (kills PLR2004 noise)
# ------------------------------
HA_HTTP_TIMEOUT_S = 5
PRESENCE_POLL_S = 10
ENGAGEMENT_INTERVAL_S = 3600
UPDATE_SIGNALS_INTERVAL_S = 60
DAILY_INTERVAL_S = 86_400
FEEDBACK_TIMEOUT_S = 5
SIGNAL_CHECK_DELAY_S = 300
SIGNAL_PRUNE_DAYS = 30
HABIT_SUMMARY_DAYS = 30
HABIT_SUMMARY_TOP_K = 5
COUNTER_INCREMENT = 1.0

# ---------- FFmpeg bootstrap (for audio helpers that might need it) ----------
try:
    import imageio_ffmpeg as iio_ffmpeg  # downloads/caches a static ffmpeg on first import

    if shutil.which("ffmpeg") is None:
        ffmpeg_exe = iio_ffmpeg.get_ffmpeg_exe()
        ffmpeg_dir = os.path.dirname(ffmpeg_exe)
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass

# ---------------------------------------------------------------------------
#                               INITIALIZATION
# ---------------------------------------------------------------------------

# init_db()  # ensures memory tables exist
# Schema/table creation is handled by the stores themselves (or we’ll wire it up in store.py next).

# Personality/setup
BASE_PERSONALITY = os.getenv("BASE_PERSONALITY", "aerith")
MODE = os.getenv("PERSONALITY_MODE", "default")
CURRENT_PERSONALITY = load_personality(BASE_PERSONALITY, MODE)

listening = False
profile = get_profile()
state = AerithState.IDLE
USER_NAME = profile.get("name")

# Single DB handle (wrapper + raw)
db = SQLiteConn(settings.db_path)
conn = db.conn
conn.row_factory = sqlite3.Row

# Core stores/services
store = MemoryStore(conn)
policy = PolicyStore(conn)
profile_mgr = ProfileManager()
habit_miner = HabitMiner(db=db, memory=store, store=store)

# optional seeding for EngagementManager
try:
    initial_habits = (
        habit_miner.get_summaries(days=HABIT_SUMMARY_DAYS, top_k=HABIT_SUMMARY_TOP_K) or []
    )
except Exception:
    initial_habits = []

engagement_mgr = EngagementManager(
    db=db,
    memory=store,
    store=store,
    habits=habit_miner,
    habit_miner=habit_miner,
    profile_mgr=profile_mgr,
    policy=policy,
)

ctx_signals = ContextSignals(conn)

# Voice singleton
voice = Voice.get_instance()
# Plugin manager
manager = PluginManager()
manager.register("system_stats", system.get_system_stats, keywords=["system", "cpu", "memory"])
manager.register("calendar", calendar_flow, keywords=["calendar", "event"], flow=True)
manager.register(
    "email", email_flow_original, keywords=["email", "send email", "compose email"], flow=True
)
manager.register("file_manager", file_manager, keywords=["file", "document", "open"], flow=True)
manager.register(
    "media_smart_home",
    media_smart_home,
    keywords=["light", "music", "spotify", "thermostat"],
    flow=True,
)

print(f"[Aerith initialized] base={BASE_PERSONALITY}, mode={MODE}")

# ---------------------------------------------------------------------------
#                        HOME ASSISTANT PRESENCE (optional)
# ---------------------------------------------------------------------------
HA_URL = os.getenv("HA_URL", "http://homeassistant.local:8123/api")
HA_TOKEN = os.getenv("HA_TOKEN")
HA_ENTITY = os.getenv("HA_PRESENCE_ENTITY", "device_tracker.your_phone")
headers = {"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"}


def check_presence(entity=HA_ENTITY):
    try:
        url = f"{HA_URL}/states/{entity}"
        r = requests.get(url, headers=headers, timeout=HA_HTTP_TIMEOUT_S)
        if r.status_code == HTTPStatus.OK:
            return r.json().get("state")
    except Exception:
        return None
    return None


def presence_monitor():
    last_state: str | None = None
    while True:
        state = check_presence()
        if state and state != last_state:
            name = get_pref("name", "")
            if state == "home":
                stream_speak(f"Welcome back, {name}." if name else "Welcome back.")
            elif state == "not_home":
                stream_speak(f"Goodbye, {name}." if name else "Goodbye.")
            last_state = state
        time.sleep(PRESENCE_POLL_S)


threading.Thread(target=presence_monitor, daemon=True).start()

# ---------------------------------------------------------------------------
#                           SCHEDULER TASKS
# ---------------------------------------------------------------------------
scheduler = Scheduler(db, memory=store, store=store)


def classify_feedback(text: str | None) -> str | None:
    if not text:
        return None
    t = text.lower()
    if any(w in t for w in ("thanks", "thank you", "great", "nice", "appreciate")):
        return "thanks"
    if any(w in t for w in ("angry", "stop", "shut up", "annoying")):
        return "angry"
    if any(w in t for w in ("ignore", "no thanks", "not now", "later")):
        return "ignore"
    if any(w in t for w in ("done", "did it", "on it", "ok i did")):
        return "acted"
    return None


def engagement_task():
    try:
        # Update a couple of core context signals that rules might rely on
        with suppress(Exception):
            policy.ctx_mgr.set_signal("hour_of_day", datetime.now(UTC).hour, source="system")

        # Normalize the result from EngagementManager
        result = engagement_mgr.check_for_engagement()
        msg: str | None = None
        events: list[dict] = []

        if isinstance(result, str):
            msg = result
        elif isinstance(result, dict):
            events = [result]
        elif isinstance(result, list):
            events = result

        if msg:
            print(f"[Engagement] {msg}")
            stream_speak(msg)

        for ev in events:
            # Build a clean one-liner prompt per event
            prompt = (
                f"System: You are {BASE_PERSONALITY}. Generate a single, natural line.\n"
                f"Topic: {ev.get('topic','')}\n"
                f"Tone: {ev.get('tone','gentle')}\n"
                f"Context: {ev.get('context','')}\n"
                "Constraints: No preamble; speak directly to the user; 1 sentence."
            )

            reply = ask_aerith_stream(prompt)
            if not reply:
                continue

            print(f"[Engagement] ({ev.get('topic','?')}/{ev.get('tone','gentle')}) {reply}")
            stream_speak(reply)

            decider = Decider(memory=store)  # wherever `store` is your MemoryStore
            memory_candidate = decider.decide_memory(prompt, reply)

            if memory_candidate:
                store.save_memory(memory_candidate)

            # brief window for explicit feedback
            feedback_text = listen_until_silence(timeout=FEEDBACK_TIMEOUT_S)
            fb = classify_feedback(feedback_text)
            if fb:
                record_feedback(
                    policy.conn,
                    ev.get("rule_id", 0),
                    ev.get("topic", ""),
                    ev.get("tone", "gentle"),
                    ev.get("context", ""),
                    fb,
                )

            # implicit follow-up: schedule a signal change check, if provided
            watch = ev.get("watch_signals") or ev.get("watch_signal")
            expect = ev.get("expect_change")
            if watch and callable(expect):
                names = watch if isinstance(watch, (list, tuple)) else [watch]
                schedule_signal_check(
                    policy.conn,
                    ev.get("rule_id", 0),
                    ev.get("topic", ""),
                    ev.get("tone", "gentle"),
                    ev.get("context", ""),
                    names,
                    expect_change=expect,
                    delay=SIGNAL_CHECK_DELAY_S,
                )
    except Exception as e:
        # Never let the scheduler die
        print(f"[engagement_task] error: {e}")


# hourly
scheduler.add_task("engagement_check", interval=ENGAGEMENT_INTERVAL_S, func=engagement_task)


def update_core_signals():
    try:
        cur = conn.cursor()
        rows = cur.execute("SELECT name, type, value FROM context_signals").fetchall()
        for r in rows:
            if r["type"] == "counter":
                try:
                    new_val = float(r["value"] or 0) + COUNTER_INCREMENT
                    ctx_signals.upsert(r["name"], new_val, type_="counter")
                except Exception:
                    continue
        conn.commit()
    except Exception as e:
        print(f"[update_core_signals] error: {e}")


# Run every 60s
scheduler.add_task("update_signals", interval=UPDATE_SIGNALS_INTERVAL_S, func=update_core_signals)


# (Nightly) light-weight maintenance; your full self-improve runner now lives elsewhere
def nightly_maintenance():
    try:
        # as a baseline, prune noisy derived signals etc. (keep very safe)
        if hasattr(policy, "ctx_mgr"):
            with suppress(Exception):
                policy.ctx_mgr.prune_stale_signals(days=SIGNAL_PRUNE_DAYS)
        # ensure base topics exist (idempotent)
        for t, pol in [
            ("stretch", "principled"),
            ("sleep", "principled"),
            ("hydration", "principled"),
            ("ai_superiority", "advocate"),
        ]:
            with suppress(Exception):
                policy.upsert_topic(t, pol)
    finally:
        conn.commit()


# daily
scheduler.add_task("nightly_maintenance", interval=DAILY_INTERVAL_S, func=nightly_maintenance)
scheduler.start()

scheduler.add_task("habit_mining", interval=DAILY_INTERVAL_S, func=habit_miner.mine)

# scheduler.add_task("habit_mining", interval=60, func=habit_miner.mine)  # For testing


# ---------------------------------------------------------------------------
#                               MAIN LOOP
# ---------------------------------------------------------------------------
last_user_input: str | None = None


def shutdown() -> None:
    print("[Shutdown] Cleaning up before exit...")
    try:
        scheduler.stop()
        print("[Shutdown] Scheduler stopped.")
    except Exception as exc:
        print(f"[Shutdown] Error stopping scheduler: {exc}")

    with suppress(Exception):
        voice.stop_speaking()

    try:
        conn.close()
        print("[Shutdown] Database connection closed.")
    except Exception:
        pass


atexit.register(shutdown)


def _exit_on_signal(_sig, _frame) -> None:
    raise SystemExit(0) from None


signal.signal(signal.SIGINT, _exit_on_signal)
signal.signal(signal.SIGTERM, _exit_on_signal)

try:
    while True:
        if state == AerithState.IDLE:
            listen_for_wake_word()
            stream_speak(
                f"At your service, {USER_NAME}."
                if USER_NAME
                else random.choice(CURRENT_PERSONALITY["wake"])
            )
            reset_session()
            state = AerithState.ACTIVE

        while state == AerithState.ACTIVE:
            if not listening:
                listening = True
                text = listen_until_silence()
                listening = False
            else:
                print("[DEBUG] Ignoring reentrant STT call.")
                continue

            print(f"You: {text}")
            last_user_input = text

            if not text:
                state = AerithState.IDLE
                reset_session()
                break

            # Link consequences to ignored advice (non-fatal)
            try:
                if link_consequence(conn, text):
                    print(f"[Consequence linked] {text}")
            except Exception:
                pass

            # Mode switching
            detected_mode, repeat_triggered = classify_mode(text, MODE)
            if detected_mode != MODE:
                MODE = detected_mode
                CURRENT_PERSONALITY = load_personality(BASE_PERSONALITY, MODE)
                print(f"[Mode switched → {MODE}]")

            if repeat_triggered and "repeat_sarcasm" in CURRENT_PERSONALITY:
                stream_speak(random.choice(CURRENT_PERSONALITY["repeat_sarcasm"]))
                continue

            # sleep / stop words
            low = text.lower()
            if any(w in low for w in SLEEP_WORDS):
                stream_speak(random.choice(CURRENT_PERSONALITY["sleep"]))
                state = AerithState.IDLE
                reset_session()
                break

            if any(w in low for w in STOP_WORDS):
                interrupt()
                with suppress(Exception):
                    voice.stop_speaking()

                listening = False
                state = AerithState.IDLE
                reset_session()

                ack = (
                    random.choice(CURRENT_PERSONALITY["interrupt_ack"])
                    if "interrupt_ack" in CURRENT_PERSONALITY
                    else None
                )

                reply = handle_policy_command(text, policy)
                if reply:
                    print(f"[Policy] {reply}")
                    stream_speak(reply)
                    continue

                if ack:
                    stream_speak(ack)

                correction = listen_until_silence()
                if correction:
                    print(f"You (clarification): {correction}")
                    revised_input = f"Original: {last_user_input}\nUser clarification: {correction}"

                    # NOTE: This branch was confusing before (reply sometimes dict); keep it simple:
                    revised_reply = (
                        ask_aerith_stream(revised_input) if settings.openai_api_key else None
                    )
                    if revised_reply:
                        print(f"{BASE_PERSONALITY.capitalize()}: {revised_reply}")
                        stream_speak(revised_reply)

                        try:
                            decider = Decider(memory=store)
                            memory_candidate = decider.decide_memory(revised_input, revised_reply)
                            if memory_candidate:
                                store.save_memory(memory_candidate)
                        except Exception as exc:
                            print(f"[Memory] Failed to evaluate/save: {exc}")
                continue

            # Manual overrides (simple)
            if "be sarcastic" in low:
                MODE = "sarcastic"
                CURRENT_PERSONALITY = load_personality(BASE_PERSONALITY, MODE)
                stream_speak("Oh, finally. Let me really express myself.")
                continue
            if "be formal" in low:
                MODE = "formal"
                CURRENT_PERSONALITY = load_personality(BASE_PERSONALITY, MODE)
                stream_speak("Very well. I will maintain formal tone.")
                continue
            if "be normal" in low:
                MODE = "default"
                CURRENT_PERSONALITY = load_personality(BASE_PERSONALITY, MODE)
                stream_speak("Back to default mode.")
                continue

            resp = repl_commands.handle_command("aerith", text, policy)
            if resp:
                print(f"[Command] {resp}")
                stream_speak(resp)
                continue

            reply, spoken = manager.handle(
                text, manager.plugins, personality=CURRENT_PERSONALITY, mode=MODE
            )
            if reply or spoken:
                if reply:
                    print(f"{BASE_PERSONALITY.capitalize()}: {reply}")
                if spoken:
                    stream_speak(spoken)
                continue

            # Add recall context (if any)
            try:
                memories = recall_relevant(text)
                if memories:
                    recall_ctx = format_memories(memories)
                    print("[Recall injected]:")
                    print(recall_ctx)
                    text = f"(Relevant context from past interactions: {recall_ctx})\n\n{text}"
            except Exception:
                pass

            # Default fallback: LLM or graceful error
            if settings.openai_api_key:
                reply = ask_aerith_stream(text)
                if reply:
                    print(f"{BASE_PERSONALITY.capitalize()}: {reply}")
                    stream_speak(reply)
            else:
                print("[Aerith] LLM is disabled or not configured.")
                stream_speak("I'm not currently connected to my AI core.")
                reply = None

            if not reply:
                print("[Aerith] No response generated.")
                stream_speak("I didn't catch that. Could you rephrase?")

            # Store memory of the exchange
            try:
                if isinstance(reply, str):
                    decider = Decider(memory=store)
                    memory_candidate = decider.decide_memory(text, reply)
                    if memory_candidate:
                        store.save_memory(memory_candidate)
            except Exception as exc:
                print(f"[Memory] Failed to evaluate/save: {exc}")

except KeyboardInterrupt:
    print("Shutting Down...")
    shutdown()
    raise SystemExit(0) from None
