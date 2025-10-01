
# base/plugins/__init__.py
from .system import get_system_stats
from base.core.profile import get_pref
from base.calendar.calendar import (
    get_upcoming_events,
    add_event,
    calendar_has_pending,
    init_google_calendar_service,
)
from .gmail import get_unread_emails, send_email, init_gmail_service

def show_calendar() -> str:
    default_cal = str(get_pref("default_calendar", "primary"))
    events = get_upcoming_events(calendar_id=default_cal, n=5)
    if not events:
        return "No upcoming events."
    return ", ".join(f"{e['summary']} on {e['start']}" for e in events)

PLUGINS = {
    "system_stats": get_system_stats,
    "spotify_control": lambda: "Spotify control not yet implemented.",
    "toggle_light": lambda: "Light toggled (stub).",
    "calendar": show_calendar,      # ✅ returns a string now
    # "add_event": add_event,       # requires text; leave out or wrap separately
    "email": get_unread_emails,
    "send_email": send_email,
}

# Initialize Google services (optional)
try:
    init_google_calendar_service()
except Exception:
    pass

try:
    init_gmail_service()
except Exception:
    pass
