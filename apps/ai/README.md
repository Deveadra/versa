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

## Runtime-facing test strategy (WS22 / issue #100)

`apps/ai` is tested as an HTTP runtime boundary, not only as internal helper logic.

- Route-level contract tests in [`server.test.ts`](./src/server.test.ts) boot an ephemeral server and exercise:
  - health/capabilities surfaces
  - legacy bridge health/capabilities/invoke behavior
  - `/ai/execute` status mapping for legacy and TypeScript paths
  - `/skills/execute` approval and error-path behavior
- Assertions focus on externally visible contracts: status codes, response shape, and policy outcomes.
- External boundaries are faked only via existing package seams and environment configuration (bridge mode, approval flags), avoiding broad rewrites.

This keeps confidence focused on `@versa/ai` as the control-plane request/response boundary while preserving existing ownership boundaries.
