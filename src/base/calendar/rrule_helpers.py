import re

WEEKDAYS = {
    "monday": "MO",
    "tuesday": "TU",
    "wednesday": "WE",
    "thursday": "TH",
    "friday": "FR",
    "saturday": "SA",
    "sunday": "SU",
}


def _time_to_hms(s: str):
    # '10am', '10:30', '10:30am'
    s = s.strip().lower()
    ampm = None
    if s.endswith("am") or s.endswith("pm"):
        ampm = s[-2:]
        s = s[:-2]
    parts = s.split(":")
    h = int(parts[0])
    m = int(parts[1]) if len(parts) > 1 else 0
    if ampm == "pm" and h != 12:
        h += 12
    if ampm == "am" and h == 12:
        h = 0
    return h, m, 0


def rrule_from_phrase(phrase: str, dtstart_iso: str | None = None) -> str | None:
    """
    Very simple phrase → RRULE converter.
    Supported:
      - 'every monday at 10am'
      - 'daily at 7:30'
      - 'weekly on friday at 15:00'
      - 'monthly on the 15th at 9am'
    """
    p = phrase.strip().lower()

    # daily at X
    m = re.search(r"daily at ([0-9:apm]+)", p)
    if m:
        h, mi, s = _time_to_hms(m.group(1))
        return f"FREQ=DAILY;BYHOUR={h};BYMINUTE={mi};BYSECOND={s}"

    # every <weekday> at X
    m = re.search(
        r"every (monday|tuesday|wednesday|thursday|friday|saturday|sunday) at ([0-9:apm]+)", p
    )
    if m:
        byday = WEEKDAYS[m.group(1)]
        h, mi, s = _time_to_hms(m.group(2))
        return f"FREQ=WEEKLY;BYDAY={byday};BYHOUR={h};BYMINUTE={mi};BYSECOND={s}"

    # weekly on <weekday> at X
    m = re.search(
        r"weekly on (monday|tuesday|wednesday|thursday|friday|saturday|sunday) at ([0-9:apm]+)", p
    )
    if m:
        byday = WEEKDAYS[m.group(1)]
        h, mi, s = _time_to_hms(m.group(2))
        return f"FREQ=WEEKLY;BYDAY={byday};BYHOUR={h};BYMINUTE={mi};BYSECOND={s}"

    # monthly on the 15th at X
    m = re.search(r"monthly on the (\d+)(?:st|nd|rd|th)? at ([0-9:apm]+)", p)
    if m:
        bymonthday = int(m.group(1))
        h, mi, s = _time_to_hms(m.group(2))
        return f"FREQ=MONTHLY;BYMONTHDAY={bymonthday};BYHOUR={h};BYMINUTE={mi};BYSECOND={s}"

    return None
