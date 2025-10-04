# base/calendar/calendar.py
import datetime
import os

import dateparser

try:
    import pickle

    from google.auth.transport.requests import Request
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    GAPI_AVAILABLE = True
except ImportError:
    GAPI_AVAILABLE = False

mock_calendar: list[dict] = []  # In-memory list of events (fallback)
_pending = None  # shape: {"summary": str|None, "time": datetime|None}

SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/gmail.modify",
]

_service = None


def init_google_calendar_service():
    """Authenticate and build Google Calendar service."""
    global _service
    token_path = "token.google.pickle"
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
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, "wb") as token:
            pickle.dump(creds, token)
    _service = build("calendar", "v3", credentials=creds)
    return _service


def calendar_has_pending():
    return _pending is not None


def get_upcoming_events(calendar_id: str = "primary", n: int = 5) -> list[dict[str, str]]:
    """
    Return the next n events as a list of dicts with keys:
      - 'summary': str
      - 'start'  : str (ISO datetime or YYYY-MM-DD)
    Always returns a list (possibly empty). No strings.
    """
    if _service:
        # NOTE: Google API param is 'calendarId' (camelCase), not 'calendar_id'
        events_result = (
            _service.events()
            .list(calendarId=calendar_id, maxResults=n, singleEvents=True, orderBy="startTime")
            .execute()
        )
        items = events_result.get("items", [])
        result: list[dict[str, str]] = []
        for e in items:
            start = e.get("start", {})
            when = start.get("dateTime") or start.get("date") or ""
            result.append({"summary": e.get("summary", ""), "start": when})
        return result
    else:
        if not mock_calendar:
            return []
        events = sorted(mock_calendar, key=lambda e: e["time"])[:n]
        return [
            {"summary": e["summary"], "start": e["time"].strftime("%Y-%m-%d %H:%M")} for e in events
        ]


def _extract_summary(natural_text: str) -> str:
    summary = natural_text
    for token in [
        "add",
        "event",
        "schedule",
        "on",
        "at",
        "for",
        "tomorrow",
        "today",
        "next",
        "this",
        "coming",
        "me",
        "to",
        "a",
        "an",
    ]:
        summary = summary.replace(token, "")
    summary = " ".join(summary.split())
    return summary.title()


def add_event(natural_text: str, calendar_id: str = "primary") -> str:
    """Parse a natural language request and add an event."""
    global _pending

    parsed_time = dateparser.parse(natural_text, settings={"PREFER_DATES_FROM": "future"})
    summary = _extract_summary(natural_text)

    if not parsed_time or not summary:
        return "I need both a time and a title for the event. Could you clarify?"

    if _service:
        event_body = {
            "summary": summary,
            "start": {"dateTime": parsed_time.isoformat(), "timeZone": "UTC"},
            "end": {
                "dateTime": (parsed_time + datetime.timedelta(hours=1)).isoformat(),
                "timeZone": "UTC",
            },
        }
        # NOTE: camelCase here too
        event = _service.events().insert(calendarId=calendar_id, body=event_body).execute()
        when = event["start"].get("dateTime") or event["start"].get("date")
        return f"Event '{event['summary']}' created at {when}"
    else:
        event = {"summary": summary, "time": parsed_time}
        mock_calendar.append(event)
        return f"Event '{event['summary']}' scheduled for {event['time'].strftime('%Y-%m-%d %H:%M')} (mock)."
