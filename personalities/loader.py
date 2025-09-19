import json
import os

# Base folder for personalities
PERSONALITIES_DIR = os.path.join(os.path.dirname(__file__))

# Cache to avoid reloading from disk repeatedly
_loaded_personalities = {}

def load_personality(base: str, mode: str = "default"):
    """
    Load a specific personality base (e.g., 'jarvis', 'ultron') and mode
    (e.g., 'default', 'sarcastic', 'formal').

    Returns a dictionary of voicelines.
    """
    key = f"{base}_{mode}"
    if key in _loaded_personalities:
        return _loaded_personalities[key]

    path = os.path.join(PERSONALITIES_DIR, base, f"{mode}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Personality file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    _loaded_personalities[key] = data
    return data


def list_available_bases():
    """
    Return all available bases (folders) under personalities.
    """
    return [
        name for name in os.listdir(PERSONALITIES_DIR)
        if os.path.isdir(os.path.join(PERSONALITIES_DIR, name))
    ]


def list_modes(base: str):
    """
    List available modes for a given base.
    """
    base_path = os.path.join(PERSONALITIES_DIR, base)
    if not os.path.exists(base_path):
        return []
    return [
        os.path.splitext(f)[0]
        for f in os.listdir(base_path)
        if f.endswith(".json")
    ]
