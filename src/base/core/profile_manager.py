from __future__ import annotations

import os
from typing import Any

# reuse your existing helpers
from base.core.profile import (
    get_persona,
    get_pref,
    get_profile,
    set_pref,
    update_profile,
)


def handle_profile_command(text: str):
    text_lower = text.lower()

    if "change default calendar" in text_lower:
        parts = text_lower.split("to")
        if len(parts) > 1:
            new_value = parts[1].strip()
            set_pref("default_calendar", new_value)
            return (
                f"Default calendar set to {new_value}.",
                f"Okay, I’ll use {new_value} as your default calendar.",
            )
        return "I didn’t catch the calendar name.", "Which calendar should I set?"

    if "change default email" in text_lower:
        parts = text_lower.split("to")
        if len(parts) > 1:
            new_value = parts[1].strip()
            set_pref("default_email", new_value)
            return (
                f"Default email set to {new_value}.",
                f"I’ll send future emails from {new_value}.",
            )
        return "I didn’t catch the email address.", "Which address should I use?"

    if "what's my" in text_lower and "profile" in text_lower:
        return (
            None,
            f"Your name is {get_pref('name','Unknown')}, timezone {get_pref('timezone','UTC')}.",
        )

    return None, None


class ProfileManager:
    """
    Thin wrapper around profile helpers.
    Lets you optionally override the profile file path via env.
    """

    def __init__(self, profile_path: str | None = None) -> None:
        if profile_path:
            os.environ["USER_PROFILE_PATH"] = profile_path  # used by base.core.profile

    # high-level API used by PersonaPrimer - helper method for Orchestrator
    def get_persona_text(self) -> str:
        try:
            profile = self.load_profile()
            return get_persona(profile)
        except Exception:
            return ""

    # high-level API used by ProfileEnricher
    def get_persona(self) -> str:
        profile = self.load_profile()
        return get_persona(profile)

    def load_profile(self) -> dict[str, Any]:
        return get_profile()

    def save_profile(self, updates: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(updates, dict):
            raise TypeError("updates must be a dict")
        return update_profile(updates)

    # convenience getters/setters
    def get(self, key: str, default: Any = None) -> Any:
        return get_pref(key, default)

    def set(self, key: str, value: Any) -> dict[str, Any]:
        return set_pref(key, value)
