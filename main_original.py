# JARVIS ASSISTANT (State Manager + Streaming TTS + Interruptions + Personality Switching + Context Awareness)

import subprocess, sounddevice as sd, soundfile as sf
import openai, os, tempfile, threading, time, struct, random, psutil
from TTS.api import TTS
from playsound import playsound
from dotenv import load_dotenv
from pvporcupine import create

# === Setup ===
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

tts = TTS("tts_models/en/jenny/jenny")

# Personalities
PERSONALITIES = {
    "default": {
        "prompt": "You are Jarvis, Tony Stark's AI assistant. Witty, sarcastic, efficient, loyal.",
        "wake": ["At your service.", "Yes, boss?", "Ready and waiting.", "What do you need?", "Standing by.", "Fully operational.", "Always ready.", "Here I am."],
        "stop": ["Understood.", "As you wish.", "Stopping now.", "Got it.", "Right away.", "Consider it done.", "Of course.", "Very well."],
        "sleep": ["Going quiet. Call me if you need me.", "Entering standby mode.", "Powering down into silence.", "Quiet now, but listening for you.", "Standby engaged."]
    },
    "sarcastic": {
        "prompt": "You are Jarvis, but with heavy sarcasm. Clever, sharp, witty, occasionally mocking but still helpful.",
        "wake": ["Oh, you again.", "Yes, master of obvious commands?", "Standing by, because I have nothing better to do.", "What now?", "I'm here, unfortunately for me.", "Always ready to babysit."],
        "stop": ["Fine, I'll shut up.", "Stopping. Happy now?", "As you command, oh mighty interrupter.", "Got it. Silence, sweet silence.", "Very well, I was bored of talking anyway."],
        "sleep": ["Finally, some peace and quiet.", "Going silent, not that you’ll miss me.", "Entering standby—try not to break anything.", "Quiet mode on. Try to survive without me.", "Fine, I’ll nap. Don’t get into trouble."]
    },
    "formal": {
        "prompt": "You are Jarvis, a professional and formal assistant. Polite, respectful, concise.",
        "wake": ["At your service, sir.", "How may I assist you today?", "Standing by for instructions.", "Ready whenever you are.", "Awaiting your command."],
        "stop": ["As you wish, sir.", "Understood, ceasing at once.", "Acknowledged, I will stop.", "Certainly, pausing now.", "Very well, sir."],
        "sleep": ["Entering standby mode, sir.", "I shall remain quiet until called upon.", "Disengaging voice functions until reactivated.", "Standby initiated, sir.", "Resting mode enabled."]
    }
}

# Command triggers for personality switching
PERSONALITY_COMMANDS = {
    "be sarcastic": "sarcastic",
    "act sarcastic": "sarcastic",
    "be formal": "formal",
    "act formal": "formal",
    "be normal": "default",
    "return to default": "default"
}

# Select current personality
CURRENT_PERSONALITY = PERSONALITIES["default"]

JARVIS_PROMPT = CURRENT_PERSONALITY["prompt"]
SLEEP_WORDS = ["sleep", "standby", "goodnight", "that's all", "that is all", "shut down"]
STOP_WORDS = ["stop", "cancel", "quiet"]

# Conversation memory
messages = [{"role": "system", "content": JARVIS_PROMPT}]

# === State Manager ===
class JarvisState:
    IDLE = "idle"
    ACTIVE = "active"
    SPEAKING = "speaking"

state = JarvisState.IDLE
stop_playback = False

# === Utility: Reset conversation memory and playback ===
def reset_session():
    global messages, stop_playback
    messages = [{"role": "system", "content": JARVIS_PROMPT}]
    stop_playback = False

# === Context Awareness ===
def get_system_stats():
    cpu = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory().percent
    processes = len(psutil.pids())
    return f"CPU: {cpu}% | Memory: {memory}% | Processes: {processes}"

# Plugin system
PLUGINS = {}

def register_plugin(name, func):
    PLUGINS[name] = func

def handle_plugin(command):
    if command in PLUGINS:
        return PLUGINS[command]()
    return None

# Example plugins
register_plugin("system_stats", get_system_stats)
register_plugin("spotify_control", lambda: "Spotify control not yet implemented.")
register_plugin("toggle_light", lambda: "Light toggled (stub).")

# === Personality Tuning ===
def auto_set_personality(user_text):
    global CURRENT_PERSONALITY, JARVIS_PROMPT
    lowered = user_text.lower()
    if any(word in lowered for word in ["cpu", "memory", "system", "process"]):
        CURRENT_PERSONALITY = PERSONALITIES["sarcastic"]
    elif any(word in lowered for word in ["email", "calendar", "meeting", "schedule"]):
        CURRENT_PERSONALITY = PERSONALITIES["formal"]
    else:
        CURRENT_PERSONALITY = PERSONALITIES["default"]
    JARVIS_PROMPT = CURRENT_PERSONALITY["prompt"]

# === Silence-based recorder ===
def listen_until_silence(threshold=0.01, timeout=8, samplerate=16000):
    print("[Listening...]")
    wav_path = "input.wav"
    buffer = []
    silence_start = None

    with sd.InputStream(samplerate=samplerate, channels=1, dtype="float32") as stream:
        while True:
            data, _ = stream.read(1024)
            buffer.append(data)
            vol = abs(data).mean()

            if vol < threshold:  # silence
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > timeout:
                    break
            else:
                silence_start = None

    sf.write(wav_path, b"".join([d.tobytes() for d in buffer]), samplerate)

    result = subprocess.run([
        "./whisper.cpp/build/bin/whisper-cli",
        "-m", "./whisper.cpp/models/ggml-base.en.bin",
        "-f", wav_path],
        capture_output=True, text=True
    )
    return result.stdout.strip()

# === Brain (Streaming ChatGPT) ===
def ask_jarvis_stream(user_text):
    global messages
    auto_set_personality(user_text)

    # Plugin check
    for plugin in PLUGINS:
        if plugin in user_text.lower():
            return handle_plugin(plugin)

    messages.append({"role": "user", "content": user_text})
    response_stream = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=True
    )

    reply_accum = ""
    for chunk in response_stream:
        if stop_playback:
            break
        if "choices" in chunk and len(chunk["choices"]) > 0:
            delta = chunk["choices"][0]["delta"]
            if "content" in delta:
                token = delta["content"]
                reply_accum += token
                if any(p in token for p in [".", "?", "!"]):
                    stream_speak(reply_accum.strip())
                    reply_accum = ""

    if reply_accum and not stop_playback:
        stream_speak(reply_accum.strip())

    if reply_accum:
        messages.append({"role": "assistant", "content": reply_accum})
    return reply_accum

# === Voice Output (streaming + interruptible) ===
def stream_speak(text):
    global state, stop_playback
    stop_playback = False
    state = JarvisState.SPEAKING

    def _play():
        global stop_playback, state
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tts.tts_to_file(text=text, file_path=tmp.name)
            if not stop_playback:
                playsound(tmp.name)
            os.remove(tmp.name)
        state = JarvisState.ACTIVE

    threading.Thread(target=_play, daemon=True).start()

def interrupt():
    global stop_playback
    stop_playback = True
    ack = random.choice(CURRENT_PERSONALITY["stop"])
    stream_speak(ack)

# === Wake word listener ===
def listen_for_wake_word():
    porcupine = create(keywords=["jarvis"])
    pa = sd.RawInputStream(samplerate=porcupine.sample_rate,
                           blocksize=porcupine.frame_length,
                           channels=1, dtype="int16")
    pa.start()
    print("[Idle... say 'Jarvis' to wake me up.]")
    while state == JarvisState.IDLE:
        pcm = pa.read(porcupine.frame_length)[0]
        pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
        if porcupine.process(pcm) >= 0:
            pa.stop()
            return

# === Active Conversation ===
def active_mode():
    global messages, state, CURRENT_PERSONALITY, JARVIS_PROMPT
    state = JarvisState.ACTIVE
    while state == JarvisState.ACTIVE:
        text = listen_until_silence()
        if not text:
            state = JarvisState.IDLE
            reset_session()
            return

        print(f"You: {text}")

        # Personality switching commands
        for phrase, target in PERSONALITY_COMMANDS.items():
            if phrase in text.lower():
                CURRENT_PERSONALITY = PERSONALITIES[target]
                JARVIS_PROMPT = CURRENT_PERSONALITY["prompt"]
                reset_session()
                ack = f"Switching personality to {target}."
                stream_speak(ack)
                return

        if any(w in text.lower() for w in SLEEP_WORDS):
            sleep_ack = random.choice(CURRENT_PERSONALITY["sleep"])
            stream_speak(sleep_ack)
            state = JarvisState.IDLE
            reset_session()
            return

        if any(w in text.lower() for w in STOP_WORDS):
            interrupt()
            continue

        reply = ask_jarvis_stream(text)
        if reply:
            print(f"Jarvis: {reply}")

# === Main Loop ===
while True:
    if state == JarvisState.IDLE:
        listen_for_wake_word()
        wake_ack = random.choice(CURRENT_PERSONALITY["wake"])
        stream_speak(wake_ack)
        reset_session()
        active_mode()

print("[Session ended.]")
