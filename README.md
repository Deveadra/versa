# Quickstart


# Aerith MVP (Local, Modular, Upgradeable) 

### Notes

- This MVP is intentionally conservative on memory writes to avoid bloat.
- Replace the simple in‑memory vector cache with FAISS/pgvector when you cross ~50k memories.
- Add scheduled tasks to consolidate older events into summaries.
- Knowledge Graph can be layered without breaking this API: mirror facts/events into entities/relations.
- Current dimension = 384 (MiniLM). Adjust in `Orchestrator` if you use a different model.
- If you expect >1M memories, consider FAISS with an IVF or HNSW index for scalability.

> This repo uses **pyproject.toml** as the source of truth for dependencies.

Install extras as needed:

- `.[dev]` → pytest + tooling (ruff/black/pylint/etc.)
- `.[voice]` → voice/TTS-related deps (optional)
- `.[google]` → Google integrations (optional)
- `.[api]` → API-related deps (optional)

---

## Prerequisites

- Python **3.11.x**
- Git
- (Optional) Any system deps required by your chosen extras (voice/google/api)

Verify Python:
```bash
python --version
```

# Clean Rebuild

## Automation

Linux/WSL/macOS: 
```bash
sudo apt update
sudo apt install -y python3-venv
./scripts/bootstrap.sh
```

Windows: `.\scripts\bootstrap.ps1`

---

## Manual

> Windows (PowerShell)

From the repo root:

```powershell
Remove-Item -Recurse -Force .venv -ErrorAction SilentlyContinue

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"

Copy-Item .env.sample .env -ErrorAction SilentlyContinue
# If .env.sample does not exist, create .env manually.

pytest -q
python run.py
```

---

> macOS / Linux / WSL
```bash
rm -rf .venv
python3 --version
python3 -m venv .venv
source .venv/bin/activate

python --version
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"

cp .env.sample .env 2>/dev/null || true
# If .env.sample does not exist, create .env manually.

pytest -q
python run.py
```

---

# Environment variables (.env)

Create `.env` in the repo root.

Minimum common keys:

- `OPENAI_API_KEY` (enables LLM features)

> Optional keys depend on extras you install (`voice`, `google`, `api`).

# Install Options (extras)

## Dev install (recommended)
```bash
python -m pip install -e ".[dev]"
```

## Optional extras (add only what you need)
```bash
python -m pip install -e ".[dev,voice]"
python -m pip install -e ".[dev,google]"
python -m pip install -e ".[dev,api]"
```

## Prod install (no dev tooling)
```bash
python -m pip install -e .
```

---

## Run
```bash
python run.py
```

## Testing

### Fast/default
```bash
pytest -q
```

### Database test suite
```bash
pytest -q tests/database -ra
```

### Diagnostics (example: disable pytest-randomly if it’s installed)
```bash
pytest tests/database -ra -p no:randomly
```

Write CI-friendly test reports into the repo
```bash
mkdir -p reports/pytest

pytest tests/database \
  -ra \
  --junitxml=reports/pytest/junit.xml \
  --cov=base \
  --cov-report=term \
  --cov-report=html:reports/pytest/htmlcov \
  --cov-report=xml:reports/pytest/coverage.xml \
  --log-file=reports/pytest/pytest.log \
  --log-level=INFO
```

---

# Linting / Formatting

> All tools below are installed by: python -m pip install -e ".[dev]"
> Installed by `python -m pip install -e ".[dev]"`

### Ruff (lint + autofix)
```bash
ruff check . --fix
```

### Black (format)
```bash
black .
```

### Pylint (optional if you want it)

Run against your source directory (adjust if your code lives elsewhere):
```bash
pylint src
```

Generate a config file:
```bash
pylint --generate-rcfile > .pylintrc
```

Optional: Write a report file
```bash
mkdir -p reports/pylint
pylint src > reports/pylint/pylint_report.txt
```

---

# CI (recommended)

CI should install from `pyproject.toml` (avoid mixing reqs files):
```YAML
- name: Install dependencies
  run: |
    python -m pip install --upgrade pip
    python -m pip install -e ".[dev]"

- name: Test
  run: pytest -q

- name: Lint
  run: |
    ruff check .
    black --check .
    pylint src
```

---

Requirements `*.txt` wrappers (optional)

If you kept wrapper files, treat them as convenience only.
They should effectively map back to pyproject extras (same end result).

Examples:
```bash
pip install -r requirements-dev.txt     # should equal: pip install -e ".[dev]"
# pip install -r requirements-voice.txt # should equal: pip install -e ".[voice]"
# pip install -r requirements.txt       # should equal: pip install -e "."
```

## 2) Design Notes

SQLite for durable memory. Facts (upsert) vs Events (append). TTL pruning.

Selective memory via assess_importance threshold.

Semantic retrieval hooks via Embeddings + in‑memory index (swap to FAISS/pgvector later).

LLM brain is pluggable. Default: OpenAI Chat. Replace with local LLM serving as needed.

Voice (ElevenLabs) and Home Assistant are optional modules.

## 3) Upgrades (Where to extend)

Swap embeddings: aerith/embeddings/provider.py

Swap vector index: build FAISS or pgvector and wire in retrieval.py

Add KG: create aerith/kg/ and sync entities from facts/events

Add background jobs: summarization, consolidation, pruning

Add tools/skills: aerith/devices/ plugins

---

### Modes
Aerith can run in different modes, controlled by `.env`:
- `AERITH_MODE=text` → text REPL (default)
- `AERITH_MODE=voice` → mic input + STT (Whisper) + TTS (ElevenLabs)
- `AERITH_MODE=stream` → future live conversation mode

### Semantic Search with FAISS

- Aerith now uses **FAISS** for vector similarity (inner product / cosine).
- All new events ingested via `Retriever.index_texts()` will be encoded and stored in FAISS.
- At bootstrap, the last 500 events are indexed.
- Keyword FTS fallback still exists for resilience.

### Vector Backend Abstraction

- Aerith now uses a `VectorBackend` interface.
- Current default: **FAISSBackend** (local, fast, MVP-friendly).
- Future backends: WeaviateBackend, MilvusBackend — swap in with a config change.

### Consolidation Jobs

- Added `Consolidator` to summarize/prune older low-importance events.
- Periodically run `summarize_old_events()` (e.g., cron, asyncio task) to:
- Merge hundreds of stale entries into a single high-level note.
- Keep vector index sharp.
- Prevent DB bloat over years.

### Automatic Consolidation (Cron)

- Aerith now runs a **daily consolidation job** at 03:00 server time.
- Uses APScheduler, running in the background.
- Summarizes and prunes older events into concise notes.
- Configurable: adjust hour/minute in `orchestrator.py` when adding job.
- Configure via `.env`:
- `AERITH_CONSOLIDATION_HOUR` (0–23)
- `AERITH_CONSOLIDATION_MINUTE` (0–59)
- Example: run every day at 1:30 AM:
```env
AERITH_CONSOLIDATION_HOUR=1
AERITH_CONSOLIDATION_MINUTE=30
```

### Knowledge Graph Reasoning (MVP)

- Aerith now uses the KG automatically when queries suggest entity/relationship questions.
- Aerith now extracts **entities** and **relations** from events.
- Entities and relations stored in SQLite tables (`entities`, `relations`).
- Simple NER via spaCy; relation hints are rule-based (expandable).
- Detection is naive (keywords: who, relation, related, about). Future: improved NER + multi-hop reasoning.
- Auto-ingests on every new memory.
- Example query:
```python
kg = KGStore(SQLiteConn("./aerith.db"))
print(kg.query_relations("Alice"))
# → [("Alice", "has_sibling", "User")]
```

### Multi-Hop Traversal
- Aerith can now follow relation chains in the KG.

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

### How rules get into the DB (seed via SQL or commands)
We don’t hardcode them. You (or Aerith, during “dream”) insert them. Example seeds (optional, purely data).

- If you don’t like these, delete/disable them (enabled=0)—no code changes needed.
- Aerith can write new rows during the dream job based on correlations (below):

```sql
INSERT INTO engagement_rules
(name, topic_id, priority, cooldown_seconds, max_per_day, condition_json, tone_strategy_json, context_template)
VALUES
(
  'rule.sedentary_break',
  'movement',
  40, 1800, 6,
  '{"cond":{"gte":[{"signal":"sedentary_minutes"},120]},
    "severity":{"between":[{"signal":"sedentary_minutes"},90,240]},
    "bindings":{"sitting_minutes":{"signal":"sedentary_minutes"}}}',
  '{"map":[{"gte":["severity",0.7],"tone":"firm"},{"gte":["severity",0.4],"tone":"persistent"}],"default":"gentle"}',
  "sitting_minutes={{sitting_minutes}}"
);

INSERT INTO engagement_rules
(name, topic_id, priority, cooldown_seconds, max_per_day, condition_json, tone_strategy_json, context_template)
VALUES
(
  'rule.hydration_gap',
  'hydration',
  50, 3600, 6,
  '{"cond":{"gte":[{"signal":"minutes_since_water"},120]},
    "severity":{"between":[{"signal":"minutes_since_water"},90,240]},
    "bindings":{"gap":{"signal":"minutes_since_water"}}}',
  '{"map":[{"gte":["severity",0.7],"tone":"firm"},{"gte":["severity",0.4],"tone":"persistent"}],"default":"gentle"}',
  "gap={{gap}}"
);
```

## Mentor Mode

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