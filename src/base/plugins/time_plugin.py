import datetime

import pytz

from base.core.profile import get_pref


def handle_time_command(text):
    if "time" in text.lower():
        tz_name = str(get_pref("timezone", "UTC") or "UTC")
        try:
            tz = pytz.timezone(tz_name)
        except Exception:
            tz = pytz.UTC
        now = datetime.datetime.now(tz)
        spoken = f"It is {now.strftime('%I:%M %p')} in {tz.zone}."
        return None, spoken
    return None, None
