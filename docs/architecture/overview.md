# Architecture overview

- Local-first core service persists to SQLite.
- Web app consumes core API.
- AI adapter is optional; core must function without it.
- Shared package owns canonical schemas and events.
