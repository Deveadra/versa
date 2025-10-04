# base/plugins/__init__.py
from base.calendar.calendar import (
    add_event,
    calendar_has_pending,
    get_upcoming_events,
    init_google_calendar_service,
)
from base.core.profile import get_pref

from .gmail import get_unread_emails, init_gmail_service, send_email
from .system import get_system_stats


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
    "calendar": show_calendar,  # ✅ returns a string now
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
