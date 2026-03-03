
### How you use it

In your running text UI:

**Type:**
```bash
propose: switch memory search to FTS triggers and LIKE fallback, and add a guard to store.py keyword_search to strip commas from MATCH query
```

**Aerith:**

- Scans codebase

- Drafts a proposal (edits base/memory/store.py, maybe tests)

- Applies local changes

- Creates branch aerith/proposal/...

- Commits, pushes, opens a PR — prints the PR URL

- Logs an event you can query later

### Security & guardrails

- Only writes files under allowlist.

- Caps patch size and file count.

- Uses anchor-replace when possible, to avoid deleting unrelated code blocks.

- No secrets are generated or written.

--- 
---
---


## Summary
- Cron schedule now configurable via `.env`.
- Defaults to **03:00**, but easy to adjust without code changes.
- Supports flexible deployment (local, cloud, or container).
- Auto-updates from memory events.
- SQLite backend now, upgrade path to Neo4j/Weaviate.
- Allows structured queries and reasoning (bridge to knowledge graph reasoning).

---


To change the daily consolidation time without touching code, add these to your `.env` and restart Aerith:


```env
# Scheduler (24h format)
AERITH_CRON_HOUR=3
AERITH_CRON_MINUTE=0
```

# CLI Commands

```bash
# Full repo, just check (default)
python scripts/diagnostic_scan.py

# Scan only changed files (working tree + untracked)
python scripts/diagnostic_scan.py --changed

# Diff against a base ref (e.g., main), plus working tree changes
python scripts/diagnostic_scan.py --changed --base origin/main

# Apply fixes (format & lint auto-fix) on all files
python scripts/diagnostic_scan.py --all --fix

# Apply fixes, but only to changed files
python scripts/diagnostic_scan.py --changed --fix
```

---

# TODO List

1. Extend your nightly job to adjust rules and invent new ones from observed outcomes:
Outcomes (“acted/thanks/ignore/angry”) get written by you wherever you capture user behavior, via:

```bash
policy.conn.execute("INSERT INTO rule_history(rule_id, topic_id, tone, context, outcome) VALUES(?,?,?,?,?)",
                    (rule_id, topic, tone, context, outcome))
# Update EMAs in rule_stats accordingly (your feedback hook)
```

2. In Dream Cycle Code

When creating or refining rules:
```bash
cur.execute("INSERT OR IGNORE INTO topics (id) VALUES (?)", (topic_id,))
```

So if Aerith spawns a new rule about "screen_time", that topic is automatically registered.

3. Hook Mood to Tone Memory

When engagement manager prepares context:

```bash
tone = choose_tone_for_topic(self.db, row["topic_id"])
last_example = style_complaint(cluster_row["last_example"], mood=tone) if cluster_row else None
```

4. Review Options

Voice: “Aerith, approve rule 2.” → status='approved'.

Text: “Show me pending rules.” → he reads from DB.

Logs: Check logs/morning_review_YYYYMMDD.txt.

CLI (optional):

python manage.py review --list
python manage.py review --approve 3
python manage.py review --deny 4

5. Design Notes

SQLite for durable memory. Facts (upsert) vs Events (append). TTL pruning.

Selective memory via assess_importance threshold.

Semantic retrieval hooks via Embeddings + in‑memory index (swap to FAISS/pgvector later).

LLM brain is pluggable. Default: OpenAI Chat. Replace with local LLM serving as needed.

Voice (ElevenLabs) and Home Assistant are optional modules.

6. Upgrades (Where to extend)

Swap embeddings: aerith/embeddings/provider.py

Swap vector index: build FAISS or pgvector and wire in retrieval.py

Add KG: create aerith/kg/ and sync entities from facts/events

Add background jobs: summarization, consolidation, pruning

Add tools/skills: aerith/devices/ plugins