
import re
from datetime import datetime
from dateutil import parser as dateparser

def extract_time_from_text(text: str):
    """
    Extracts a datetime or year/month from text.
    Returns (start_iso, end_iso) or (None, None) if nothing found.
    """
    # Simple year match
    m = re.search(r"(19|20)\\d{2}", text)
    if m:
        year = int(m.group())
        start = datetime(year, 1, 1)
        end = datetime(year, 12, 31, 23, 59, 59)
        return start.isoformat(), end.isoformat()

    # Look for full dates or months
    try:
        dt = dateparser.parse(text, fuzzy=True, default=datetime.utcnow())
        # If only month/year given
        if dt.day == datetime.utcnow().day:  # fallback default day inserted
            start = datetime(dt.year, dt.month, 1)
            if dt.month == 12:
                end = datetime(dt.year, 12, 31, 23, 59, 59)
            else:
                end = datetime(dt.year, dt.month + 1, 1)
        else:
            # full date provided
            start = datetime(dt.year, dt.month, dt.day)
            end = datetime(dt.year, dt.month, dt.day, 23, 59, 59)
        return start.isoformat(), end.isoformat()
    except Exception:
        return None, None
