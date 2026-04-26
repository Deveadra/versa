# WS02 — Telemetry and observability foundation

Issue: `https://github.com/Deveadra/versa/issues/47`

## Objective

Establish a shared, additive telemetry foundation so `apps/core` and `apps/ai` emit consistent structured events with traceability metadata.

## Telemetry contract surface

Canonical telemetry contracts are defined in `@versa/shared`:

- `TraceContextSchema`
  - `traceId` (required)
  - `correlationId` (optional)
  - `runId` (optional)
  - `requestId` (optional)
  - `parentTraceId` (optional)
- `TelemetryActorSchema`
  - `service`
  - `source`
  - `actorId` (optional)
- `TelemetryEventSchema`
  - `eventId`
  - `eventType`
  - `level`
  - `message`
  - `timestamp`
  - `actor`
  - `context`
  - `attributes`

These contracts are importable across apps/packages and are intended to remain stable for downstream workstreams.

## Logging package foundation

`@versa/logging` now provides:

- `createLogger(options)`
  - emits validated structured telemetry events
  - supports actor defaults and inherited trace context
- `createRequestTelemetryMiddleware(logger)`
  - emits `http.request.started` and `http.request.completed`
  - propagates/assigns trace metadata from request headers
- `createNdjsonFileSink(path)`
  - appends telemetry events as newline-delimited JSON (durable local sink)
- legacy compatibility API: `log(level, message, data)`

## Traceability conventions

- `traceId`
  - primary per-request/per-operation linkage key.
  - accepted from `x-trace-id` header when present.
- `correlationId`
  - cross-request grouping key (for multi-request workflows).
  - accepted from `x-correlation-id` header when present.
- `runId`
  - long-running workflow/session identifier.
  - accepted from `x-run-id` header when present.
- `requestId`
  - per-request identifier.
  - accepted from `x-request-id` header; generated if absent.

## Baseline app wiring

- `apps/core`
  - request lifecycle middleware enabled
  - domain event logging emits structured telemetry alongside existing event persistence
  - startup/shutdown telemetry events added
- `apps/ai`
  - request lifecycle middleware enabled
  - startup/shutdown telemetry events added

## Durable telemetry output strategy (current)

- Console output remains available when `TELEMETRY_CONSOLE_ENABLED=true`.
- Durable sink writes NDJSON to `artifacts/telemetry.ndjson` when `TELEMETRY_ENABLED=true`.
- This provides a simple, vendor-neutral persistence baseline for debugging/auditing.

## Current boundaries

- No operator UI/dashboard work is included here.
- No vendor-specific telemetry backend integration is required for this slice.
- No approvals/memory/MCP subsystem implementation is introduced in WS02.
