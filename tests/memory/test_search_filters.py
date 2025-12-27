
from datetime import datetime
import time


def test_search_since_filter(memstore):
    memstore.add_event("old thing", importance=1.0, type_="event")

    # cutoff is "now" AFTER the old event
    cutoff = datetime.utcnow().isoformat()

    # ensure new event timestamp is after cutoff (tiny sleep for safety)
    time.sleep(0.01)

    memstore.add_event("new thing", importance=1.0, type_="event")

    hits = memstore.search("thing", since=cutoff, limit=10)
    assert any("new thing" in h for h in hits)
    assert all("old thing" not in h for h in hits)
