# Copilot Instructions for AI Coding Agents

## Project Overview
- **Ultron** is a modular, local-first AI assistant with pluggable memory, LLM, and skills.
- Core data flow: user actions → memory/events (SQLite) → semantic retrieval (FAISS/pgvector) → LLM reasoning → response.
- Major components:
  - `base/agents/orchestrator.py`: Main control loop, schedules jobs, routes requests.
  - `base/core/`: Context, memory, plugin, and profile management.
  - `base/embeddings/`: Embedding providers and vector index abstraction.
  - `base/kg/`: Knowledge graph (entities, relations, traversal).
  - `base/learning/`: Habit mining, profile enrichment, feedback.
  - `base/database/`: SQLite backend, migrations.
  - `base/devices/`: Home Assistant and device plugins.
  - `apps/`: Prompt personalities and app-specific logic.

## Key Patterns & Conventions
- **Memory**: Facts (upsert) vs Events (append-only). Use `assess_importance` for selective retention.
- **Semantic Search**: Default is in-memory; swap to FAISS/pgvector for scale in `embeddings/provider.py`.
- **Knowledge Graph**: Entities/relations auto-extracted from events; stored in SQLite. Traverse via `kg/store.py`.
- **Scheduled Jobs**: Daily consolidation (summarize/prune) via APScheduler, configured in `.env` (`ULTRON_CONSOLIDATION_HOUR`, `ULTRON_CONSOLIDATION_MINUTE`).
- **Personality/Profiles**: User habits and preferences mined from `usage_log` and updated in `user_profile.json`.
- **Extensibility**: Add new skills/tools in `devices/` or `plugins/`. Add new embedding/vector backends in `embeddings/`.

## Developer Workflows
- **Setup**: `python -m venv .venv; .venv\Scripts\activate; pip install -r requirements.txt`
- **Run**: `python run.py` (uses `.env` for config)
- **Test**: See `tests/` for examples. Use `pytest` or run test files directly.
- **Migrations**: Place new SQL files in `database/migrations/`; orchestrator runs all on startup.
- **Modes**: Set `ULTRON_MODE` in `.env` (`text`, `voice`, `stream`).

## Integration Points
- **LLM**: Default is OpenAI; swap in local LLM by editing orchestrator/provider config.
- **Voice**: ElevenLabs/Whisper optional; see `io/voice.py` and `.env`.
- **Home Automation**: Integrate via `devices/home_assistant.py`.

## Examples
- Add a new device: create a module in `devices/`, register in plugin manager.
- Add a new embedding backend: implement `VectorBackend` in `embeddings/provider.py`.
- Add a new habit: extend `HabitMiner.update_from_usage()` in `learning/habits.py`.

## Tips
- Keep memory lean: consolidate/prune old events.
- Use semantic hooks for retrieval, not just keyword search.
- Reference `README.md` for up-to-date architecture and extension notes.

---

For more, see `README.md` and comments in orchestrator, embeddings, and learning modules.
