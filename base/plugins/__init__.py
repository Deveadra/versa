from .system import get_system_stats
from .calendar import get_upcoming_events, add_event, calendar_has_pending, init_google_calendar_service
from .gmail import get_unread_emails, send_email, init_gmail_service

PLUGINS = {
    "system_stats": get_system_stats,
    "spotify_control": lambda: "Spotify control not yet implemented.",
    "toggle_light": lambda: "Light toggled (stub).",
    "calendar": get_upcoming_events,
    "add_event": add_event,
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
