
from pathlib import Path
from dotenv import load_dotenv
import os

from base.memory import decider

# Always resolve path relative to this file (main.py)
dotenv_path = Path(__file__).parent / "config" / ".env"
load_dotenv(dotenv_path=dotenv_path)

import random
import threading
import time
import requests
import sqlite3


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
from base.memory.decider import Decider
from base.memory.store import init_db, save_memory
from base.memory.recall import recall_relevant, format_memories
from base.memory.store import MemoryStore
from config.config import settings

init_db()

    
# ===================== Personality Config =====================
BASE_PERSONALITY = os.getenv("BASE_PERSONALITY", "ultron")   # default to Ultron
MODE = os.getenv("PERSONALITY_MODE", "default")              # default mode

CURRENT_PERSONALITY = load_personality(BASE_PERSONALITY, MODE)

profile = get_profile()
USER_NAME = profile.get("name", None)


# ===================== Plugin Manager =====================
manager = PluginManager()
manager.register("system_stats", system.get_system_stats, keywords=["system", "cpu", "memory"])
manager.register("calendar", calendar_flow, keywords=["calendar", "event"], flow=True)
manager.register("email", email_flow_original, keywords=["email", "send email", "compose email"], flow=True)
manager.register("file_manager", file_manager, keywords=["file", "document", "open"], flow=True)
manager.register("media_smart_home", media_smart_home, keywords=["light", "music", "spotify", "thermostat"], flow=True)
# manager.register("profile", profile_manager, keywords=["profile", "preferences", "settings"], flow=True)


state = JarvisState.IDLE
conn = sqlite3.connect(settings.db_path, check_same_thread=False)
store = MemoryStore(conn)

print(f"[Jarvis initialized with base personality: {BASE_PERSONALITY}, mode: {MODE}]")
print(os.getenv("PICOVOICE_API_KEY"))


# ===================== Presence Monitor =====================
HA_URL = os.getenv("HA_URL", "http://homeassistant.local:8123/api")
HA_TOKEN = os.getenv("HA_TOKEN")
HA_ENTITY = os.getenv("HA_PRESENCE_ENTITY", "device_tracker.your_phone")

headers = {"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"}

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
            if not text:
                state = JarvisState.IDLE
                reset_session()
                break

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
                interrupt()
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
