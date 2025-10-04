import json
import os

PROFILE_PATH = os.getenv("USER_PROFILE_PATH", "config/user_profile.json")


def get_profile():
    """
    Load user profile from disk. Returns a dict.
    """
    try:
        with open(PROFILE_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}


def get_persona(profile: dict) -> str:
    """
    Extract a persona description from the profile.
    """
    name = profile.get("name", "User")
    profession = profile.get("profession", "a professional")
    traits = profile.get("traits", [])
    interests = profile.get("interests", [])

    traits_str = ", ".join(traits) if traits else "no specific traits"
    interests_str = ", ".join(interests) if interests else "no specific interests"

    persona = (
        f"{name} is {profession} with {traits_str}. " f"They are interested in {interests_str}."
    )
    return persona


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


def set_pref(key: str, value):
    """
    Set a single preference and persist it to disk.
    Returns the updated profile dict.
    """
    if not isinstance(key, str) or not key:
        raise ValueError("Preference key must be a non-empty string.")

    # (Optional) ensure it's JSON-serializable early
    try:
        json.dumps({key: value})
    except TypeError as e:
        raise TypeError(f"Preference value for '{key}' must be JSON-serializable: {e}")

    return update_profile({key: value})
