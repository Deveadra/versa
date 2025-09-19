import os
import requests
import random
import time

from base.core.profile import get_pref
from base.core.audio import stream_speak
from base.devices.home_assistant import call_service, get_state
from base.apps.media_smart_home_prompts import ASK_DEVICE_VARIANTS, ASK_ACTION_VARIANTS, CONFIRM_VARIANTS, CANCEL_VARIANTS


HA_URL = os.getenv("HA_URL", "http://homeassistant.local:8123/api")
HA_TOKEN = os.getenv("HA_TOKEN")

headers = {"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"}

media_state = {"device": None, "action": None, "confirm": False}

    
def handle_media_command(text):
    text_lower = text.lower()

    # TV control
    if "turn on tv" in text_lower:
        call_service("media_player", "turn_on", {"entity_id": "media_player.living_room_tv"})
        return None, "Turning on the TV."
    if "turn off tv" in text_lower:
        call_service("media_player", "turn_off", {"entity_id": "media_player.living_room_tv"})
        return None, "Turning off the TV."

    # Spotify control
    if "play" in text_lower and "spotify" in text_lower:
        query = text_lower.replace("play", "").replace("spotify", "").strip()
        # For simplicity: assume playlist/track IDs mapped manually in HA or config
        call_service("media_player", "play_media", {
            "entity_id": "media_player.spotify",
            "media_content_type": "music",
            "media_content_id": query
        })
        return None, f"Playing {query} on Spotify."

    if "pause music" in text_lower or "stop music" in text_lower:
        call_service("media_player", "media_pause", {"entity_id": "media_player.spotify"})
        return None, "Music paused."

    return None, None


# --- HA Helpers ---
def check_presence(entity="device_tracker.your_phone"):
    url = f"{HA_URL}/states/{entity}"
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        return r.json()["state"]
    return None

def presence_monitor():
    last_state = None
    while True:
        state = check_presence()
        if last_state != state:
            if state == "home":
                stream_speak("Welcome home, sir. Shall I turn on the lights?")
            elif state == "not_home":
                stream_speak("Goodbye. I’ll keep things ready until you return.")
            last_state = state
        time.sleep(10)  # poll every 10s
        
def call_service(domain, service, entity_id=None, data=None):
    """Generic helper to call HA services."""
    url = f"{HA_URL}/services/{domain}/{service}"
    payload = {"entity_id": entity_id} if entity_id else {}
    if data:
        payload.update(data)
    r = requests.post(url, headers=headers, json=payload)
    return r.status_code == 200, r.text


def control_lights(action, entity_id="light.living_room"):
    if action in ["on", "turn on", "switch on"]:
        ok, msg = call_service("light", "turn_on", entity_id)
        return "Lights turned on." if ok else f"Failed: {msg}"
    elif action in ["off", "turn off", "switch off"]:
        ok, msg = call_service("light", "turn_off", entity_id)
        return "Lights turned off." if ok else f"Failed: {msg}"
    return "I’m not sure what to do with the lights."


def control_music(action, entity_id="media_player.spotify"):
    if action in ["play", "resume", "start"]:
        ok, msg = call_service("media_player", "media_play", entity_id)
        return "Music is now playing." if ok else f"Failed: {msg}"
    elif action in ["pause", "stop"]:
        ok, msg = call_service("media_player", "media_pause", entity_id)
        return "Music paused." if ok else f"Failed: {msg}"
    elif action in ["next", "skip"]:
        ok, msg = call_service("media_player", "media_next_track", entity_id)
        return "Skipped to next track." if ok else f"Failed: {msg}"
    return "I’m not sure what to do with the music."


def control_thermostat(action, entity_id="climate.living_room"):
    if "heat" in action:
        ok, msg = call_service("climate", "set_hvac_mode", entity_id, {"hvac_mode": "heat"})
        return "Thermostat set to heat." if ok else f"Failed: {msg}"
    elif "cool" in action:
        ok, msg = call_service("climate", "set_hvac_mode", entity_id, {"hvac_mode": "cool"})
        return "Thermostat set to cool." if ok else f"Failed: {msg}"
    elif "off" in action:
        ok, msg = call_service("climate", "set_hvac_mode", entity_id, {"hvac_mode": "off"})
        return "Thermostat turned off." if ok else f"Failed: {msg}"
    return "I’m not sure what to do with the thermostat."


# --- Conversation Flow ---
def has_pending():
    return any(v is None for k, v in media_state.items() if k != "confirm") or media_state["confirm"]


def is_media_command(text: str) -> bool:
    return any(word in text.lower() for word in ["music", "spotify", "light", "thermostat"])


def handle(text: str, active_plugins):
    global media_state

    # Device step
    if media_state["device"] is None:
        if any(d in text.lower() for d in ["music", "spotify"]):
            media_state["device"] = "music"
        elif "light" in text.lower():
            media_state["device"] = "lights"
        elif "thermostat" in text.lower():
            media_state["device"] = "thermostat"
        else:
            return None, random.choice(ASK_DEVICE_VARIANTS)
        return None, random.choice(ASK_ACTION_VARIANTS).format(device=media_state["device"])

    # Action step
    if media_state["action"] is None:
        media_state["action"] = text.strip().lower()
        media_state["confirm"] = True
        return None, f"You want me to {media_state['action']} the {media_state['device']}?"

    # Confirmation step
    if media_state["confirm"]:
        if text.lower() in ["yes", "confirm", "do it"]:
            if media_state["device"] == "lights":
                result = control_lights(media_state["action"])
            elif media_state["device"] == "music":
                result = control_music(media_state["action"])
            elif media_state["device"] == "thermostat":
                result = control_thermostat(media_state["action"])
            else:
                result = "Unknown device."
            media_state.update({"device": None, "action": None, "confirm": False})
            return result, random.choice(CONFIRM_VARIANTS)
        else:
            media_state.update({"device": None, "action": None, "confirm": False})
            return "Smart Home action cancelled.", random.choice(CANCEL_VARIANTS)

    # Initial trigger
    if is_media_command(text):
        media_state.update({"device": None, "action": None, "confirm": False})
        return None, random.choice(ASK_DEVICE_VARIANTS)

    return None, None
