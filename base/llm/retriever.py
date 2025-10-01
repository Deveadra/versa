
from __future__ import annotations
from typing import List, Dict, Any, Optional
import math, datetime

HALF_LIFE_DAYS = 90.0

class Retriever:
    def __init__(self, conn):
        self.conn = conn

    def _recency_boost(self, ts: Optional[str]) -> float:
        if not ts:
            return 0.0
        try:
            dt = datetime.datetime.fromisoformat(ts)
        except Exception:
            try:
                dt = datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            except Exception:
                return 0.0
        delta = (datetime.datetime.utcnow() - dt).total_seconds() / 86400.0
        return math.exp(-math.log(2) * (delta / HALF_LIFE_DAYS))

    def _score_fact_row(self, row, query_terms: List[str]) -> float:
        text = (str(row.get("key","")) + " " + str(row.get("value",""))).lower()
        matches = sum(1 for t in query_terms if t in text)
        base = min(1.0, 0.2 * matches + 0.1)
        boost = self._recency_boost(row.get("last_reinforced") or row.get("created_at"))
        return base + boost

    def _score_usage_row(self, row, query_terms: List[str]) -> float:
        text = (str(row.get("user_text","")) + " " + str(row.get("resolved_action") or "")).lower()
        matches = sum(1 for t in query_terms if t in text)
        base = min(1.0, 0.15 * matches + 0.05)
        boost = self._recency_boost(row.get("created_at"))
        return base + boost

    def query(self, user_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        q = (user_text or "").lower()
        terms = [t for t in q.split() if len(t) > 2]
        c = self.conn.cursor()
        results = []
        # facts
        try:
            c.execute("SELECT id, key, value, created_at, last_reinforced FROM facts LIMIT 500")
            rows = [dict(r) for r in c.fetchall()]
            for r in rows:
                score = self._score_fact_row(r, terms)
                if score > 0.0:
                    results.append({
                        "summary": f"fact: {r.get('key')}={str(r.get('value'))[:120]}",
                        "source": "facts",
                        "score": float(score),
                        "last_used": r.get("last_reinforced") or r.get("created_at"),
                        "created_at": r.get("created_at"),
                    })
        except Exception:
            pass
        # usage_log
        try:
            c.execute("SELECT id, user_text, resolved_action, params_json, created_at FROM usage_log ORDER BY id DESC LIMIT 1000")
            rows = [dict(r) for r in c.fetchall()]
            for r in rows:
                score = self._score_usage_row(r, terms)
                if score > 0.0:
                    results.append({
                        "summary": f"usage: {r.get('user_text')}",
                        "source": "usage_log",
                        "score": float(score),
                        "last_used": r.get("created_at"),
                        "created_at": r.get("created_at"),
                    })
        except Exception:
            pass

        results_sorted = sorted(results, key=lambda x: x.get("score", 0.0), reverse=True)
        seen = set()
        unique = []
        for r in results_sorted:
            s = r.get("summary")
            if s in seen:
                continue
            seen.add(s)
            unique.append(r)
            if len(unique) >= top_k:
                break
        return unique
