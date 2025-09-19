import datetime
import dateparser
import os

try:
    from googleapiclient.discovery import build
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    import pickle
    GAPI_AVAILABLE = True
except ImportError:
    GAPI_AVAILABLE = False

mock_calendar = []  # In-memory list of events (fallback)
_pending = None  # shape: {"summary": str|None, "time": datetime|None}

SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/gmail.modify"
]

_service = None


def init_google_calendar_service():
    """Authenticate and build Google Calendar service."""
    global _service
    token_path = 'token.google.pickle'
    if not GAPI_AVAILABLE:
        return None
    creds = None
    if os.path.exists(token_path):
      with open(token_path, "rb") as token:
          creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, "wb") as token:
            pickle.dump(creds, token)
    _service = build('calendar', 'v3', credentials=creds)
    return _service


def calendar_has_pending():
    return _pending is not None


def get_upcoming_events(n=5):
    """Return upcoming events from Google Calendar if available, else mock."""
    if _service:
        events_result = _service.events().list(
          calendarId='primary', maxResults=n, singleEvents=True,
          orderBy='startTime').execute()
        events = events_result.get('items', [])
        if not events:
            return "No events found."
        return "\n".join([f"{e['start'].get('dateTime', e['start'].get('date'))} - {e['summary']}" for e in events])
    else:
        if not mock_calendar:
            return "No events scheduled."
        events = sorted(mock_calendar, key=lambda e: e["time"])
        return "\n".join([f"{e['time'].strftime('%Y-%m-%d %H:%M')} - {e['summary']}" for e in events[:n]])


def _extract_summary(natural_text: str) -> str:
    summary = natural_text
    for token in ["add", "event", "schedule", "on", "at", "for", "tomorrow", "today", "next", "this", "coming", "me", "to", "a", "an"]:
        summary = summary.replace(token, "")
    summary = " ".join(summary.split())
    return summary.title()


def add_event(natural_text: str):
    """Parse a natural language request and add an event."""
    global _pending

    parsed_time = dateparser.parse(natural_text, settings={'PREFER_DATES_FROM': 'future'})
    summary = _extract_summary(natural_text)

    if not parsed_time or not summary:
        return "I need both a time and a title for the event. Could you clarify?"

    if _service:
        event_body = {
            'summary': summary,
            'start': {'dateTime': parsed_time.isoformat(), 'timeZone': 'UTC'},
            'end': {'dateTime': (parsed_time + datetime.timedelta(hours=1)).isoformat(), 'timeZone': 'UTC'},
        }
        event = _service.events().insert(calendarId='primary', body=event_body).execute()
        return f"Event '{event['summary']}' created at {event['start']['dateTime']}"
    else:
        event = {"summary": summary, "time": parsed_time}
        mock_calendar.append(event)
        return f"Event '{event['summary']}' scheduled for {event['time'].strftime('%Y-%m-%d %H:%M')} (mock)."