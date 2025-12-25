# main.py
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

for p in (SRC, ROOT):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
# -----------------------------------------

import atexit
import os
import random
import shutil
import signal
import sqlite3
import threading
import time

import requests
from dotenv import load_dotenv

# ---------- FFmpeg bootstrap (for audio helpers that might need it) ----------
try:
    import imageio_ffmpeg as iio_ffmpeg  # downloads/caches a static ffmpeg on first import

    if shutil.which("ffmpeg") is None:
        ffmpeg_exe = iio_ffmpeg.get_ffmpeg_exe()
        ffmpeg_dir = os.path.dirname(ffmpeg_exe)
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass

# ---------- Env ----------
dotenv_path = Path(__file__).parent / "config" / ".env"
load_dotenv(dotenv_path=dotenv_path)

# ---------- Std / 3p ----------
from datetime import datetime

from base.agents.scheduler import Scheduler
from base.calendar import calendar_flow
from base.core.audio import interrupt, listen_for_wake_word, listen_until_silence, stream_speak
from base.core.commands import handle_policy_command

# ---------- Ultron imports ----------
from base.core.core import SLEEP_WORDS, STOP_WORDS, JarvisState, reset_session
from base.core.decider import Decider
from base.core.mode_classifier import classify_mode
from base.core.plugin_manager import PluginManager
from base.core.profile import get_pref, get_profile
from base.core.profile_manager import ProfileManager
from base.database.sqlite import SQLiteConn
from base.learning.engagement_manager import EngagementManager
from base.learning.habit_miner import HabitMiner
from base.llm.brain import ask_jarvis_stream
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

# ---------------------------------------------------------------------------
#                               INITIALIZATION
# ---------------------------------------------------------------------------

# init_db()  # ensures memory tables exist
# Schema/table creation is handled by the stores themselves (or we’ll wire it up in store.py next).

# Personality/setup
BASE_PERSONALITY = os.getenv("BASE_PERSONALITY", "ultron")
MODE = os.getenv("PERSONALITY_MODE", "default")
CURRENT_PERSONALITY = load_personality(BASE_PERSONALITY, MODE)

listening = False
profile = get_profile()
state = JarvisState.IDLE
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
    initial_habits = habit_miner.get_summaries(days=30, top_k=5) or []
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

print(f"[Ultron initialized] base={BASE_PERSONALITY}, mode={MODE}")

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
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
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
        time.sleep(10)


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
        try:
            policy.ctx_mgr.set_signal("hour_of_day", datetime.utcnow().hour, source="system")
        except Exception:
            pass

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

            reply = ask_jarvis_stream(prompt)
            if not reply:
                continue

            print(f"[Engagement] ({ev.get('topic','?')}/{ev.get('tone','gentle')}) {reply}")
            stream_speak(reply)

            decider = Decider(memory=store)  # wherever `store` is your MemoryStore
            memory_candidate = decider.decide_memory(text, reply)

            if memory_candidate:
                store.save_memory(memory_candidate)

            # brief window for explicit feedback
            feedback_text = listen_until_silence(timeout=5)
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
                    delay=300,
                )
    except Exception as e:
        # Never let the scheduler die
        print(f"[engagement_task] error: {e}")


# hourly
scheduler.add_task("engagement_check", interval=3600, func=engagement_task)


def update_core_signals():
    try:
        cur = conn.cursor()
        rows = cur.execute("SELECT name, type, value FROM context_signals").fetchall()
        for r in rows:
            if r["type"] == "counter":
                try:
                    new_val = float(r["value"] or 0) + 1.0
                    ctx_signals.upsert(r["name"], new_val, type_="counter")
                except Exception:
                    continue
        conn.commit()
    except Exception as e:
        print(f"[update_core_signals] error: {e}")


# Run every 60s
scheduler.add_task("update_signals", interval=60, func=update_core_signals)


# (Nightly) light-weight maintenance; your full self-improve runner now lives elsewhere
def nightly_maintenance():
    try:
        # as a baseline, prune noisy derived signals etc. (keep very safe)
        if hasattr(policy, "ctx_mgr"):
            try:
                policy.ctx_mgr.prune_stale_signals(days=30)
            except Exception:
                pass
        # ensure base topics exist (idempotent)
        for t, pol in [
            ("stretch", "principled"),
            ("sleep", "principled"),
            ("hydration", "principled"),
            ("ai_superiority", "advocate"),
        ]:
            try:
                policy.upsert_topic(t, pol)
            except Exception:
                pass
    finally:
        conn.commit()


# daily
scheduler.add_task("nightly_maintenance", interval=86400, func=nightly_maintenance)
scheduler.start()


scheduler.add_task("habit_mining", interval=86400, func=habit_miner.mine)

# scheduler.add_task("habit_mining", interval=60, func=habit_miner.mine)  # For testing


# ---------------------------------------------------------------------------
#                               MAIN LOOP
# ---------------------------------------------------------------------------
state = JarvisState.IDLE
last_user_input: str | None = None

try:
    while True:

        def shutdown():
            print("[Shutdown] Cleaning up before exit...")
            try:
                scheduler.stop()
                print("[Shutdown] Scheduler stopped.")
            except Exception as e:
                print(f"[Shutdown] Error stopping scheduler: {e}")

            try:
                interrupt()
                voice.stop_speaking()
            except Exception:
                pass

            try:
                conn.close()
                print("[Shutdown] Database connection closed.")
            except Exception:
                pass

        # Register cleanup on exit and termination
        atexit.register(shutdown)
        signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))
        signal.signal(signal.SIGTERM, lambda sig, frame: sys.exit(0))

        while True:
            if state == JarvisState.IDLE:
                listen_for_wake_word()
                stream_speak(
                    f"At your service, {USER_NAME}."
                    if USER_NAME
                    else random.choice(CURRENT_PERSONALITY["wake"])
                )
                reset_session()
                state = JarvisState.ACTIVE

            while state == JarvisState.ACTIVE:
                # text = listen_until_silence()
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
                    state = JarvisState.IDLE
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
                    state = JarvisState.IDLE
                    reset_session()
                    break

                # if any(w in low for w in STOP_WORDS):
                #     interrupt()
                #     voice.stop_speaking()

                if any(w in low for w in STOP_WORDS):
                    interrupt()
                    try:
                        voice.stop_speaking()
                    except Exception:
                        pass

                    listening = False
                    state = JarvisState.IDLE
                    reset_session()

                    ack = None
                    if "interrupt_ack" in CURRENT_PERSONALITY:
                        ack = random.choice(CURRENT_PERSONALITY["interrupt_ack"])

                    # policy commands (e.g., “Ultron disable speak” etc.)
                    reply = handle_policy_command(text, policy)
                    if reply:
                        print(f"[Policy] {reply}")
                        stream_speak(reply)
                        continue

                    if ack:
                        stream_speak(ack)

                    # capture clarification
                    correction = listen_until_silence()
                    if correction:
                        print(f"You (clarification): {correction}")
                        revised_input = (
                            f"Original: {last_user_input}\nUser clarification: {correction}"
                        )
                        if isinstance(reply, dict):
                            reply = reply.get("response") or reply.get("content") or str(reply)

                            # reply = ask_jarvis_stream(revised_input)
                            if reply:
                                print(f"{BASE_PERSONALITY.capitalize()}: {reply}")
                                stream_speak(reply)
                                dec = Decider()
                                mem = dec.decide_memory(revised_input, reply)
                                if mem:
                                    store.add_event(
                                        f"{mem['content']} || {mem.get('response','')}",
                                        importance=0.0,
                                        type_="chat",
                                    )

                                decider = Decider(
                                    memory=store
                                )  # wherever `store` is your MemoryStore
                                memory_candidate = decider.decide_memory(text, reply)

                                if memory_candidate:
                                    store.save_memory(memory_candidate)
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

                resp = repl_commands.handle_command("ultron", text, policy)
                if resp:
                    print(f"[Command] {resp}")
                    stream_speak(resp)
                    continue

                # Plugin routing (runs before LLM fallback)
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
                    reply = ask_jarvis_stream(text)
                    if reply:
                        print(f"{BASE_PERSONALITY.capitalize()}: {reply}")
                        stream_speak(reply)
                else:
                    print("[Ultron] LLM is disabled or not configured.")
                    stream_speak("I'm not currently connected to my AI core.")
                    reply = None

                # Final fallback if no reply generated
                if not reply:
                    print("[Ultron] No response generated.")
                    stream_speak("I didn't catch that. Could you rephrase?")

                # Store memory of the exchange
                try:
                    if isinstance(reply, dict):
                        reply = reply.get("response") or reply.get("content") or str(reply)

                    if reply is not None and isinstance(reply, str):
                        decider = Decider(memory=store)
                        memory_candidate = decider.decide_memory(text, reply)

                        if memory_candidate:
                            store.save_memory(memory_candidate)
                except Exception as e:
                    print(f"[Memory] Failed to evaluate/save: {e}")

                # if settings.openai_api_key:
                #     reply = ask_jarvis_stream(text)
                #     if reply:
                #         print(f"{BASE_PERSONALITY.capitalize()}: {reply}")
                #         stream_speak(reply)
                # else:
                #     print("[Ultron] LLM is disabled or not configured.")
                #     stream_speak("I'm not currently connected to my AI core.")
                #     reply = None

                # # Final fallback if no reply generated
                # if not reply:
                #     print("[Ultron] No response generated.")
                #     stream_speak("I didn't catch that. Could you rephrase?")

                # # Store memory of the exchange
                # try:
                #     if isinstance(reply, dict):
                #         reply = reply.get("response") or reply.get("content") or str(reply)

                #     dec = Decider()
                #     mem = dec.decide_memory(text, reply or "")
                #     if mem:
                #         store.add_event(
                #             f"{mem['content']} || {mem.get('response','')}", importance=0.0, type_="chat"
                #         )

                #     decider = Decider(memory=store)  # wherever `store` is your MemoryStore
                #     memory_candidate = decider.decide_memory(text, reply)

                #     if memory_candidate:
                #         store.save_memory(memory_candidate)
                # except Exception:
                #     pass

except KeyboardInterrupt:
    print("Shutting Down...")
    scheduler.stop()
    sys.exit(0)
