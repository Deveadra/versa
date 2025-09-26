
from pathlib import Path
from dotenv import load_dotenv
import os, shutil
import datetime
try:
    import imageio_ffmpeg as iio_ffmpeg  # downloads/caches a static ffmpeg on first import

    # If ffmpeg isn't already on PATH, prepend imageio's dir so tools find it.
    if shutil.which("ffmpeg") is None:
        ffmpeg_exe = iio_ffmpeg.get_ffmpeg_exe()
        ffmpeg_dir = os.path.dirname(ffmpeg_exe)
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
except Exception:
    # Keep going; you'll catch it in the preflight check below.
    pass

from base.core import decider
from datetime import datetime

# Always resolve path relative to this file (main.py)
dotenv_path = Path(__file__).parent / "config" / ".env"
load_dotenv(dotenv_path=dotenv_path)

import random
import threading
import time
import requests
import sqlite3
import asyncio


from typing import Optional, Callable, Iterable, Dict, Any, Union


from base.core.core import JarvisState, reset_session, SLEEP_WORDS, STOP_WORDS
from base.core.audio import listen_for_wake_word, listen_until_silence, stream_speak, interrupt
from base.llm.brain import ask_jarvis_stream
from base.calendar import calendar_flow
from base.plugins import system, file_manager, media_smart_home #profile_manager
from base.core.plugin_manager import PluginManager
from personalities.loader import load_personality
from base.core.mode_classifier import classify_mode
from personalities.loader import load_personality
from base.plugins import email_flow_original
from base.core.profile import get_profile
from base.core.profile import get_pref
from base.core.decider import Decider
from base.memory.store import init_db, save_memory
from base.memory.recall import recall_relevant, format_memories
from base.memory.store import MemoryStore
from config.config import settings
from base.learning.engagement_manager import EngagementManager
from base.learning.habit_miner import HabitMiner
from base.core.profile_manager import ProfileManager
from base.policy.policy_store import PolicyStore
from base.core.commands import handle_policy_command
from base.database.sqlite import SQLiteConn
from base.agents.scheduler import Scheduler
from base.voice.tts_elevenlabs import Voice
from base.policy.feedback import record_feedback, schedule_signal_check
from base.policy.context_signals import ContextSignals
from base.policy.consequence_linker import link_consequence

# Choose your TTS engine here:
# from base.voice.tts_elevenlabs import speak_blocking as speak
# OR
from base.voice.tts_elevenlabs import speak_async as speak, stop_speaking


last_user_input = None  # track last input globally

init_db()

    
# ===================== Personality Config =====================
BASE_PERSONALITY = os.getenv("BASE_PERSONALITY", "ultron")   # default to Ultron
MODE = os.getenv("PERSONALITY_MODE", "default")              # default mode

CURRENT_PERSONALITY = load_personality(BASE_PERSONALITY, MODE)

profile = get_profile()
USER_NAME = profile.get("name", None)
conn = sqlite3.connect(settings.db_path, check_same_thread=False)
conn.row_factory = sqlite3.Row

store = MemoryStore(conn)
policy = PolicyStore(conn)   
conn = sqlite3.connect(settings.db_path, check_same_thread=False)
conn.row_factory = sqlite3.Row
habit_miner = HabitMiner(conn)
profile_mgr = ProfileManager()
engagement_mgr = EngagementManager(habit_miner, profile_mgr, policy)
scheduler = Scheduler()
state = JarvisState.IDLE
conn = sqlite3.connect(settings.db_path, check_same_thread=False)
ctx_signals = ContextSignals(conn)



# ===================== Plugin Manager =====================
manager = PluginManager()
manager.register("system_stats", system.get_system_stats, keywords=["system", "cpu", "memory"])
manager.register("calendar", calendar_flow, keywords=["calendar", "event"], flow=True)
manager.register("email", email_flow_original, keywords=["email", "send email", "compose email"], flow=True)
manager.register("file_manager", file_manager, keywords=["file", "document", "open"], flow=True)
manager.register("media_smart_home", media_smart_home, keywords=["light", "music", "spotify", "thermostat"], flow=True)
# manager.register("profile", profile_manager, keywords=["profile", "preferences", "settings"], flow=True)





print(f"[Jarvis initialized with base personality: {BASE_PERSONALITY}, mode: {MODE}]")
print(os.getenv("PICOVOICE_API_KEY"))


# ===================== Presence Monitor =====================
HA_URL = os.getenv("HA_URL", "http://homeassistant.local:8123/api")
HA_TOKEN = os.getenv("HA_TOKEN")
HA_ENTITY = os.getenv("HA_PRESENCE_ENTITY", "device_tracker.your_phone")

headers = {"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"}


# ===================== Scheduler Tasks =====================
def engagement_task():
    msg = engagement_mgr.check_for_engagement()
    ctx = policy.ctx_mgr.all_signals()
    ctx |= policy.ctx_mgr.eval_derived_signals()
    speak, meta = policy.should_speak("stretch", ctx)
    events = engagement_mgr.check_for_engagement()
    
    policy.ctx_mgr.set_signal("hour_of_day", datetime.utcnow().hour, source="system")
    
    for ev in events:
        reply = ask_jarvis_stream(prompt)
        if reply:
            speak(reply)
            # Explicit feedback via voice/text
            feedback_text = listen_until_silence(timeout=5)
            fb = classify_feedback(feedback_text) if feedback_text else None
            if fb:
                record_feedback(policy.conn, ev["rule_id"], ev["topic"], ev["tone"], ev["context"], fb)

            # Implicit feedback: schedule a check
            if ev.get("watch_signal") and ev.get("expect_change"):
                def expect_change_wrapper(current_values):
                    return ev["expect_change"](current_values)
                signal_name = ev["watch_signal"]
                def expect_change(val):
                    try:
                        return float(val) < 120  # example: sitting_minutes drops
                    except:
                        return False
                schedule_signal_check(
                policy.conn,
                ev["rule_id"],
                ev["topic"],
                ev["tone"],
                ev["context"],
                ev["watch_signals"],
                expect_change_wrapper
            )
            prompt = (
                f"System: You are {BASE_PERSONALITY}. Generate a single, natural line.\n"
                f"Topic: {ev['topic']}\nTone: {ev['tone']}\nContext: {ev['context']}\n"
                f"Constraints: No preamble; speak directly to the user; 1 sentence."
            )
            
            reply = ask_jarvis_stream(prompt)
            
            if reply:
                print(f"[Engagement] ({ev['topic']}/{ev['tone']}) {reply}")
                speak(reply)
                
                # After speaking, listen briefly for a response
                feedback_text = listen_until_silence(timeout=5)  # adjust timeout
                if feedback_text:
                    fb = classify_feedback(feedback_text)  # see next step
                    if fb:
                        from base.policy.feedback import record_feedback
                        record_feedback(policy.conn, ev["rule_id"], ev["topic"], ev["tone"], ev["context"], fb)

            if msg:
                print(f"[Engagement] {msg}")
                speak(msg)
                # later this could be routed through Ultron’s voice/text output

# Run every 3600s = 1 hour
scheduler.add_task("engagement_check", interval=3600, func=engagement_task)
scheduler.start()


def update_core_signals():
    cur = conn.cursor()
    rows = cur.execute("SELECT name, type, value FROM context_signals").fetchall()
    for r in rows:
        if r["type"] == "counter":
            try:
                new_val = float(r["value"] or 0) + 1
                ctx_signals.upsert(r["name"], new_val, type_="counter")
            except Exception:
                continue
        # future: derived signals handled here

# Run every 60s
scheduler.add_task("update_signals", interval=60, func=update_signals)


def nightly_maintenance():
    print("[Dream] Reviewing context signals...")
    policy.ctx_mgr.prune_stale_signals(days=30)
    # future: auto-generate derived signals here based on logs
    # Simple learning example: down-rank noisy rules, up-rank effective ones
    rows = policy.conn.execute("""
      SELECT r.id, r.name, r.priority, s.ema_success, s.ema_negative
      FROM engagement_rules r
      LEFT JOIN rule_stats s ON s.rule_id = r.id
      WHERE r.enabled=1
    """).fetchall()
    # === Rule self-improvement ===
    from base.policy.self_improve import propose_new_rules, insert_proposed_rules
    new_rules = propose_new_rules(policy)
    
    if new_rules:
        print(f"[Dream] Proposed {len(new_rules)} new rule(s):")
        for r in new_rules:
            print(f"  - {r['name']} (topic {r['topic_id']})")
        # Auto-insert, if you want:
        insert_proposed_rules(policy.conn, new_rules)
    else:
        print("[Dream] No new rules proposed tonight.")

    for r in rows:
        pri = r["priority"]
        suc = r["ema_success"] or 0.5
        neg = r["ema_negative"] or 0.5
        # crude heuristic: effective -> higher priority (lower number)
        new_pri = pri
        if suc > 0.65 and neg < 0.4:
            new_pri = max(1, pri - 5)
        elif neg > 0.7 and suc < 0.45:
            new_pri = min(100, pri + 10)
        if new_pri != pri:
            policy.conn.execute("UPDATE engagement_rules SET priority=?, updated_at=datetime('now') WHERE id=?",
                                (new_pri, r["id"]))
    policy.conn.commit()
    # Future: auto-create new engagement_rules by scanning common signal pairs
    # (co-occurrence analysis written to derived_signals + proposed rules table)
    
    print("[Dream] Done.")

# Run daily at ~3am (interval in seconds: 86400)
scheduler.add_task("nightly_dream", interval=86400, func=nightly_maintenance)


def check_presence(entity=HA_ENTITY):
    try:
        url = f"{HA_URL}/states/{entity}"
        r = requests.get(url, headers=headers)
        if r.status_code == 200:
            return r.json()["state"]
    except Exception:
        return None
    return None

def presence_monitor():
    last_state = None
    while True:
        state = check_presence()
        if state and state != last_state:
            name = get_pref("name", "")
            if state == "home":
                if name:
                    stream_speak(f"Welcome back, {name}.")
                else:
                    stream_speak("Welcome back.")
            elif state == "not_home":
                if name:
                    stream_speak(f"Goodbye, {name}.")
                else:
                    stream_speak("Goodbye.")
            last_state = state
        time.sleep(10) # poll every 10s

threading.Thread(target=presence_monitor, daemon=True).start()


# ===================== Main Loop =====================
while True:
    if state == JarvisState.IDLE:
        listen_for_wake_word()
        if USER_NAME:
            stream_speak(f"At your service, {USER_NAME}.")
        else:
            stream_speak(random.choice(CURRENT_PERSONALITY["wake"]))
        reset_session()
        state = JarvisState.ACTIVE

        while state == JarvisState.ACTIVE:
            text = listen_until_silence()
            print(f"You: {text}")
            last_user_input = text
            if not text:
                state = JarvisState.IDLE
                reset_session()
                break
            
            # After capturing user input
            if text:
                linked = link_consequence(conn, text)
                if linked:
                    print(f"[Consequence linked to ignored advice: {text}]")
                    
            print(f"You: {text}")

            # 🧠 Natural mode switching
            detected_mode, repeat_triggered = classify_mode(text, MODE)
            if detected_mode != MODE:
                MODE = detected_mode
                CURRENT_PERSONALITY = load_personality(BASE_PERSONALITY, MODE)
                print(f"[Mode switched to {MODE}]")

            # When speaking sarcastically after repeat
            if repeat_triggered and "repeat_sarcasm" in CURRENT_PERSONALITY:
                stream_speak(random.choice(CURRENT_PERSONALITY["repeat_sarcasm"]))
                continue

            if any(w in text.lower() for w in SLEEP_WORDS):
                stream_speak(random.choice(CURRENT_PERSONALITY["sleep"]))
                state = JarvisState.IDLE
                reset_session()
                break

            if any(w in text.lower() for w in STOP_WORDS):
                interrupt()       # cut TTS
                stop_speaking()   # stop audio playback (if using blocking TTS)

                # Pull an acknowledgment line from personality if available
                ack = None
                if "interrupt_ack" in CURRENT_PERSONALITY:
                    ack = random.choice(CURRENT_PERSONALITY["interrupt_ack"])
                reply = handle_policy_command(text, policy)
                if reply:
                    print(f"[Policy] {reply}")
                    stream_speak(reply)
                    continue

                stream_speak(ack)

                # Immediately capture clarification
                correction = listen_until_silence()
                if correction:
                    print(f"You (clarification): {correction}")

                    # Merge original input + correction for GPT
                    revised_input = (
                        f"Original question: {last_user_input}\n"
                        f"User clarification: {correction}"
                    )

                    reply = ask_jarvis_stream(revised_input)
                    if reply:
                        print(f"{BASE_PERSONALITY.capitalize()}: {reply}")
                        stream_speak(reply)

                        # Store in memory as a clarified exchange
                        decider = Decider()
                        memory = decider.decide_memory(revised_input, reply)
                        if memory:
                            store.add_event(
                                f"{memory['content']} || {memory.get('response','')}",
                                importance=0.0,
                                type_="chat"
                            )
                continue

            # (Optional) manual overrides
            if "be sarcastic" in text.lower():
                MODE = "sarcastic"
                CURRENT_PERSONALITY = load_personality(BASE_PERSONALITY, MODE)
                stream_speak("Oh, finally. Let me really express myself.")
                continue
            elif "be formal" in text.lower():
                MODE = "formal"
                CURRENT_PERSONALITY = load_personality(BASE_PERSONALITY, MODE)
                stream_speak("Very well. I will maintain formal tone.")
                continue
            elif "be normal" in text.lower():
                MODE = "default"
                CURRENT_PERSONALITY = load_personality(BASE_PERSONALITY, MODE)
                stream_speak("Back to default mode.")
                continue

            # Delegate to PluginManager
            reply, spoken = manager.handle(
                text,
                manager.plugins,
                personality=CURRENT_PERSONALITY,
                mode=MODE
            )

            if reply or spoken:
                if reply:
                    print(f"{BASE_PERSONALITY.capitalize()}: {reply}")
                if spoken:
                    stream_speak(spoken)
                continue
            
            # Inside main loop, before ask_jarvis_stream(text):
            memories = recall_relevant(text)
            if memories:
                recall_context = format_memories(memories)
                print("[Recall injected]:")
                print(recall_context)
                # Prepend to conversation so GPT sees it
                text = f"(Relevant context from past interactions: {recall_context})\n\n{text}"


            # Default fallback: GPT response
            reply = ask_jarvis_stream(text)
            if reply:
                print(f"{BASE_PERSONALITY.capitalize()}: {reply}")
                stream_speak(reply)

            decider = Decider()
            memory = decider.decide_memory(text, reply)
            if memory:
                store.add_event(f"{memory['content']} || {memory.get('response','')}", importance=0.0, type_="chat")

            for t, pol in [("stretch","principled"), ("sleep","principled"), ("hydration","principled"),
                 ("ai_superiority","advocate")]:
            policy.upsert_topic(t, pol)
