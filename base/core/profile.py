import json
import os

PROFILE_PATH = os.getenv("USER_PROFILE_PATH", "config/user_profile.json")

def get_profile():
    """
    Load user profile from disk. Returns a dict.
    """
    try:
        with open(PROFILE_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}

def update_profile(updates: dict):
    """
    Merge updates into profile and persist.
    """
    profile = get_profile()
    profile.update(updates)
    os.makedirs(os.path.dirname(PROFILE_PATH), exist_ok=True)
    with open(PROFILE_PATH, "w") as f:
        json.dump(profile, f, indent=2)
    return profile

def get_pref(key: str, default=None):
    """
    Convenience accessor for single preference.
    """
    profile = get_profile()
    return profile.get(key, default)
