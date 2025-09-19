from jarvis.core.profile import set_pref, get_pref

def handle_profile_command(text):
    text_lower = text.lower()

    if "change default calendar" in text_lower:
        parts = text_lower.split("to")
        if len(parts) > 1:
            new_value = parts[1].strip()
            set_pref("default_calendar", new_value)
            return f"Default calendar set to {new_value}.", f"Okay, I’ll use {new_value} as your default calendar."
        return "I didn’t catch the calendar name.", "Which calendar should I set?"

    if "change default email" in text_lower:
        parts = text_lower.split("to")
        if len(parts) > 1:
            new_value = parts[1].strip()
            set_pref("default_email", new_value)
            return f"Default email set to {new_value}.", f"I’ll send future emails from {new_value}."
        return "I didn’t catch the email address.", "Which address should I use?"

    if "what's my" in text_lower and "profile" in text_lower:
        return None, f"Your name is {get_pref('name','Unknown')}, timezone {get_pref('timezone','UTC')}."

    return None, None
