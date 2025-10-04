import json
import os
import random

BASE_DIR = os.path.dirname(__file__)


def load_personality(mode: str):
    """Load voicelines from the given mode's JSON file."""
    file_path = os.path.join(BASE_DIR, f"{mode}.json")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Personality file not found: {file_path}")

    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


def get_line(mode: str, category: str):
    """Return a random line from the specified mode and category."""
    data = load_personality(mode)
    if category not in data:
        raise KeyError(f"Category '{category}' not found in {mode}.json")

    choices = data[category]
    if not choices:
        return ""

    # Handle simple list of strings
    if isinstance(choices[0], str):
        return random.choice(choices)

    # Handle weighted format: [{ "line": "...", "weight": 2 }]
    weighted = []
    for entry in choices:
        weighted.extend([entry["line"]] * entry.get("weight", 1))

    return random.choice(weighted)
