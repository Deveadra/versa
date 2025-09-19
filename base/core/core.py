import random

class JarvisState:
    IDLE = "idle"
    ACTIVE = "active"
    SPEAKING = "speaking"

# Personalities
PERSONALITIES = {
    "default": {"prompt": "You are Jarvis, Tony Stark's AI assistant. Witty, sarcastic, efficient, loyal.",
        "wake": ["At your service.", "Yes, boss?"],
        "stop": ["Understood.", "As you wish."],
        "sleep": ["Going quiet. Call me if you need me."]},
    "sarcastic": {"prompt": "You are Jarvis, but with heavy sarcasm.",
        "wake": ["Oh, you again.", "Yes, master of obvious commands?"],
        "stop": ["Fine, I'll shut up.", "Stopping. Happy now?"],
        "sleep": ["Finally, some peace and quiet."]},
    "formal": {"prompt": "You are Jarvis, a professional and formal assistant.",
        "wake": ["At your service, sir.", "How may I assist you today?"],
        "stop": ["As you wish, sir.", "Understood, ceasing at once."],
        "sleep": ["Entering standby mode, sir."]}
}

PERSONALITY_COMMANDS = {
    "be sarcastic": "sarcastic",
    "act sarcastic": "sarcastic",
    "be formal": "formal",
    "act formal": "formal",
    "be normal": "default",
    "return to default": "default"
}

CURRENT_PERSONALITY = PERSONALITIES["default"]
JARVIS_PROMPT = CURRENT_PERSONALITY["prompt"]
SLEEP_WORDS = ["sleep", "standby", "goodnight"]
STOP_WORDS = ["stop", "cancel", "quiet"]

messages = [{"role": "system", "content": JARVIS_PROMPT}]

stop_playback = False
state = JarvisState.IDLE

def reset_session():
    global messages, stop_playback
    messages = [{"role": "system", "content": JARVIS_PROMPT}]
    stop_playback = False


def pick_ack(kind: str):
    return random.choice(CURRENT_PERSONALITY.get(kind, [""]))