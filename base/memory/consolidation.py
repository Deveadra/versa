
from __future__ import annotations
from datetime import datetime, timedelta
from loguru import logger
from assistant.config import settings


class Consolidator:
  def __init__(self, store, brain):
    self.store = store
    self.brain = brain


  def summarize_old_events(self):
    cutoff = (datetime.utcnow() - timedelta(days=settings.memory_ttl_days)).isoformat()
    cur = self.store.db.conn.execute("SELECT id, content FROM events WHERE ts < ? ORDER BY id LIMIT 200", (cutoff,))
    rows = cur.fetchall()
    if not rows:
      return
    texts = [r["content"] for r in rows]
    joined = "\n".join(texts)
    prompt = f"Summarize these past events into a concise knowledge note:\n{joined}"
    summary = self.brain.complete("System: consolidation", prompt, max_tokens=200)
    if summary:
      self.store.add_event(summary, importance=40, type_="summary")
      ids = [r["id"] for r in rows]
      self.store.db.conn.executemany("DELETE FROM events WHERE id=?", [(i,) for i in ids])
      self.store.db.conn.commit()
      logger.info(f"Consolidated {len(ids)} events into summary")