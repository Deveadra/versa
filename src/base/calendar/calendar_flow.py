from calendar.calendar import get_upcoming_events

from base.core.profile import get_pref
from base.core.stylizer import stylize_response

# Structured state for calendar event creation
event_state = {"title": None, "date": None, "time": None, "confirm": False}


def has_pending():
    return (
        any(v is None for k, v in event_state.items() if k != "confirm") or event_state["confirm"]
    )


def is_calendar_command(text: str) -> bool:
    return "add event" in text.lower() or "new event" in text.lower() or "calendar" in text.lower()


def handle_calendar_command(text):
    default_calendar = get_pref("default_calendar", "primary") or "primary"

    if "add" in text.lower() or "create" in text.lower():
        return None, f"Which calendar should I use? Default is {default_calendar}."

    if "show" in text.lower() or "upcoming" in text.lower():
        events = get_upcoming_events(calendar_id=str(default_calendar), n=5)
        if not events:
            return "No upcoming events.", "You have no events scheduled."
        spoken = ", ".join([f"{e['summary']} on {e['start']}" for e in events])
        return f"Found {len(events)} events.", spoken

    return None, None


def handle(text, personality=None, mode="default"):
    """
    Handle calendar-related commands.
    For now, mock some data; in real use, connect to Google Calendar.
    """

    events = [
        {"title": "Team Meeting", "time": "10:00 AM"},
        {"title": "Dentist Appointment", "time": "3:00 PM"},
    ]

    if not events:
        if personality:
            return stylize_response(personality, mode, "calendar_empty", {})
        return "No upcoming events."

    # Just return the first event for simplicity
    next_event = events[0]
    data = {"title": next_event["title"], "time": next_event["time"]}

    if personality:
        return stylize_response(personality, mode, "calendar", data)

    return f"Next event: {next_event['title']} at {next_event['time']}"


# def handle(text: str, active_plugins):
#     global event_state

#     # Title step
#     if event_state["title"] is None:
#         event_state["title"] = text.strip()
#         return None, random.choice(ASK_DATE_VARIANTS)

#     # Date step
#     if event_state["date"] is None:
#         event_state["date"] = text.strip()
#         return None, random.choice(ASK_TIME_VARIANTS)

#     # Time step
#     if event_state["time"] is None:
#         event_state["time"] = text.strip()
#         confirm_prompt = random.choice(ASK_CONFIRM_VARIANTS).format(
#             title=event_state["title"], date=event_state["date"], time=event_state["time"]
#         )
#         event_state["confirm"] = True
#         return None, confirm_prompt

#     # Confirmation step
#     if event_state["confirm"]:
#         if text.lower() in ["yes", "confirm", "add it", "do it"]:
#             reply = active_plugins["calendar"].add_event(
#                 f"{event_state['title']} on {event_state['date']} at {event_state['time']}"
#             )
#             event_state = {"title": None, "date": None, "time": None, "confirm": False}
#             return reply, random.choice(CONFIRM_EVENT_VARIANTS)
#         else:
#             event_state = {"title": None, "date": None, "time": None, "confirm": False}
#             return "Event creation cancelled.", random.choice(CANCEL_EVENT_VARIANTS)

#     # Initial trigger
#     if is_calendar_command(text):
#         event_state = {"title": None, "date": None, "time": None, "confirm": False}
#         return None, random.choice(ASK_TITLE_VARIANTS)

#    return None, None
