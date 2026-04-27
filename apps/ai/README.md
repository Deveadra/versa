# AI Adapter Service

This service defines the boundary for Ultron/Aerith capabilities.
Core calls this adapter with timeout/retry (future middleware); if unreachable, core continues without AI.

## WS10 convergence direction

`@versa/ai` is the canonical AI-facing TypeScript boundary for the redesign track.

- It keeps local TypeScript execution paths (`/skills/execute`, `/ai/execute` for non-legacy targets).
- It exposes explicit legacy bridge surfaces:
  - `GET /bridge/health`
  - `GET /bridge/capabilities`
  - `POST /bridge/invoke`
- It preserves approval-aware and telemetry-aware behavior for bridge calls.

## Ownership boundaries

- Legacy Python runtime owns legacy AI behavior execution.
- `@versa/ai` owns request shaping, policy checks, telemetry emission, and typed response shaping.
- `@versa/bridge` owns the narrow bridge adapter contract used by `@versa/ai`.

This is additive convergence work and does not replace or remove the legacy runtime.
