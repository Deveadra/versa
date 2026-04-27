# WS10 — AI service convergence and legacy Python bridge

Issue: `https://github.com/Deveadra/versa/issues/55`

## Objective

Converge `@versa/ai` as the canonical AI-facing TypeScript service boundary while adding an explicit, typed, and policy-aware bridge surface to the legacy Python runtime.

## Canonical ownership boundary

WS10 keeps responsibilities explicit and non-destructive:

- Legacy Python runtime remains the owner of legacy AI behavior execution.
- `@versa/ai` owns request boundary shaping, policy checks, telemetry events, and typed response surfaces.
- `@versa/bridge` owns a narrow adapter contract for bridge health, capability discovery, and invocation semantics.

No replacement or removal of the legacy runtime is performed in this slice.

## Shared contract surface

WS10 adds bridge and AI-convergence contracts in [`packages/shared/src/index.ts`](../../packages/shared/src/index.ts):

- `BridgeExecutionModeEnum`
- `BridgeTargetRuntimeEnum`
- `BridgeOperationEnum`
- `BridgeCapabilitySchema`
- `BridgeHealthStatusSchema`
- `BridgeRequestContextSchema`
- `LegacyBridgeRequestSchema`
- `LegacyBridgeResponseSchema`
- `AiServiceExecutionTargetEnum`
- `AiServiceRequestSchema`
- `AiServiceResponseSchema`
- `BridgeAdapterResultSchema`

These contracts make bridge operations, typed AI-service boundaries, and adapter fallback semantics importable across apps/packages.

## Config linkage

WS10 extends bridge-related config in [`packages/config/src/index.ts`](../../packages/config/src/index.ts):

- `BRIDGE_MODE` (`disabled` | `shadow` | `primary`)
- `BRIDGE_LEGACY_RUNTIME_URL`
- `BRIDGE_HEALTH_PATH`
- `BRIDGE_CAPABILITIES_PATH`
- `BRIDGE_INVOKE_PATH`

Existing bridge flags remain intact (`BRIDGE_ENABLED`, `BRIDGE_TIMEOUT_MS`, existing bridge URLs).

## Bridge package and AI runtime path

WS10 introduces `@versa/bridge` in [`packages/bridge/src/index.ts`](../../packages/bridge/src/index.ts):

- typed endpoint construction helpers
- default legacy capability declarations
- typed bridge health generation
- narrow adapter execution behavior with mode-aware fallback semantics

`@versa/ai` uses this adapter in [`apps/ai/src/server.ts`](../../apps/ai/src/server.ts) with additive routes:

- `GET /bridge/health`
- `GET /bridge/capabilities`
- `POST /bridge/invoke`
- `POST /ai/execute` (typed AI-service boundary with TypeScript-vs-legacy target routing)

Bridge invocations preserve policy and telemetry handling from prior workstreams.

## Policy and telemetry behavior

Bridge invocation in `apps/ai` remains bounded and observable:

- approval policy gate for `bridge.invoke` when `FEATURE_APPROVALS_ENABLED` is active
- structured telemetry for bridge request lifecycle:
  - `bridge.invoke.requested`
  - `bridge.invoke.completed`

This keeps bridge calls explicit, traceable, and policy-aware.

## WS10 boundaries

- No wholesale legacy Python runtime rewrite or deletion
- No collapse of all AI behavior into one oversized service
- No operator-console/frontend implementation work
- No unrelated cleanup
