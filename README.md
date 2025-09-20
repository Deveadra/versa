# Quickstart


# Ultron MVP (Local, Modular, Upgradeable) 

### Notes

- This MVP is intentionally conservative on memory writes to avoid bloat.
- Replace the simple in‑memory vector cache with FAISS/pgvector when you cross ~50k memories.
- Add scheduled tasks to consolidate older events into summaries.
- Knowledge Graph can be layered without breaking this API: mirror facts/events into entities/relations.
- Current dimension = 384 (MiniLM). Adjust in `Orchestrator` if you use a different model.
- If you expect >1M memories, consider FAISS with an IVF or HNSW index for scalability.

## 1) Setup

```bash
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.sample .env
# add your OPENAI_API_KEY, etc.
python run.py
```

## 2) Design Notes

SQLite for durable memory. Facts (upsert) vs Events (append). TTL pruning.

Selective memory via assess_importance threshold.

Semantic retrieval hooks via Embeddings + in‑memory index (swap to FAISS/pgvector later).

LLM brain is pluggable. Default: OpenAI Chat. Replace with local LLM serving as needed.

Voice (ElevenLabs) and Home Assistant are optional modules.

## 3) Upgrades (Where to extend)

Swap embeddings: ultron/embeddings/provider.py

Swap vector index: build FAISS or pgvector and wire in retrieval.py

Add KG: create ultron/kg/ and sync entities from facts/events

Add background jobs: summarization, consolidation, pruning

Add tools/skills: ultron/devices/ plugins

---

### Modes
Ultron can run in different modes, controlled by `.env`:
- `ULTRON_MODE=text` → text REPL (default)
- `ULTRON_MODE=voice` → mic input + STT (Whisper) + TTS (ElevenLabs)
- `ULTRON_MODE=stream` → future live conversation mode

### Semantic Search with FAISS

- Ultron now uses **FAISS** for vector similarity (inner product / cosine).
- All new events ingested via `Retriever.index_texts()` will be encoded and stored in FAISS.
- At bootstrap, the last 500 events are indexed.
- Keyword FTS fallback still exists for resilience.

### Vector Backend Abstraction

- Ultron now uses a `VectorBackend` interface.
- Current default: **FAISSBackend** (local, fast, MVP-friendly).
- Future backends: WeaviateBackend, MilvusBackend — swap in with a config change.

### Consolidation Jobs

- Added `Consolidator` to summarize/prune older low-importance events.
- Periodically run `summarize_old_events()` (e.g., cron, asyncio task) to:
- Merge hundreds of stale entries into a single high-level note.
- Keep vector index sharp.
- Prevent DB bloat over years.

### Automatic Consolidation (Cron)

- Ultron now runs a **daily consolidation job** at 03:00 server time.
- Uses APScheduler, running in the background.
- Summarizes and prunes older events into concise notes.
- Configurable: adjust hour/minute in `orchestrator.py` when adding job.
- Configure via `.env`:
- `ULTRON_CONSOLIDATION_HOUR` (0–23)
- `ULTRON_CONSOLIDATION_MINUTE` (0–59)
- Example: run every day at 1:30 AM:
```env
ULTRON_CONSOLIDATION_HOUR=1
ULTRON_CONSOLIDATION_MINUTE=30
```

### Knowledge Graph Reasoning (MVP)

- Ultron now uses the KG automatically when queries suggest entity/relationship questions.
- Ultron now extracts **entities** and **relations** from events.
- Entities and relations stored in SQLite tables (`entities`, `relations`).
- Simple NER via spaCy; relation hints are rule-based (expandable).
- Detection is naive (keywords: who, relation, related, about). Future: improved NER + multi-hop reasoning.
- Auto-ingests on every new memory.
- Example query:
```python
kg = KGStore(SQLiteConn("./ultron.db"))
print(kg.query_relations("Alice"))
# → [("Alice", "has_sibling", "User")]
```

### Multi-Hop Traversal
- Ultron can now follow relation chains in the KG.

### Reverse Traversal
- KG traversal now works in **both directions**.

## Learning & Personality Layer (v1)

--- 
---
---

### Data flow
1. Every user interaction/action is appended to `usage_log`.
2. `HabitMiner.update_from_usage()` scans recent usage and updates `habits` with exponential‑decay scores (half‑life 30d).
3. `ProfileEnricher.run()` derives `preferred_player`, `favorite_music`, `greeting_style`, and `sleep_time` from top habits, updating `user_profile.json`.
4. Orchestrator uses profile defaults for contextual shortcuts (e.g., `play music` → service=preferred_player, genre=favorite_music).
5. When uncertain, orchestrator asks the user to confirm. Feedback is recorded in `feedback_events` and used to tune tone policy via a small bandit.

### Extending habits
Add new key builders in `HabitMiner.update_from_usage()` to track additional domains:
- `home.lights=bright|dim`
- `calendar.reminder_minutes=10|30|60`
- `email.signature=short|detailed`

### Confidence maintenance for facts
Facts are kept fresh by periodically bumping `confidence` and `last_reinforced` when feedback affirms them; they decay implicitly with time unless reaffirmed.

### Migration
Place `0002_learning.sql` next to `0001_init.sql`. Ensure orchestrator runs both on startup.

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


To change the daily consolidation time without touching code, add these to your `.env` and restart Ultron:


```env
# Scheduler (24h format)
ULTRON_CRON_HOUR=3
ULTRON_CRON_MINUTE=0
```