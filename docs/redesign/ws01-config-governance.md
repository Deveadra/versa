# WS01 — Canonical config and dependency governance

Issue: https://github.com/Deveadra/versa/issues/46

## Canonical config surface

`@versa/config` exposes one validated runtime surface through `parseConfig(env)` and `loadConfig()` in `packages/config/src/index.ts`.

### Config groups

- App ports
  - `CORE_PORT`
  - `AI_PORT`
  - `WEB_PORT`
  - `MCP_GATEWAY_PORT`
- Runtime mode
  - `NODE_ENV` (`development | test | production`)
  - `RUNTIME_MODE` (`local | hybrid | cloud`)
- Database
  - `DATABASE_URL`
  - `DATABASE_READ_URL`
  - `DATABASE_MIGRATIONS_PATH`
- Feature flags
  - `FEATURE_MEMORY_ENABLED`
  - `FEATURE_APPROVALS_ENABLED`
  - `FEATURE_SKILLS_ENABLED`
  - `FEATURE_WORKSPACES_ENABLED`
- MCP settings
  - `MCP_ENABLED`
  - `MCP_TRANSPORT` (`stdio | http`)
  - `MCP_HOST`
  - `MCP_PORT`
- Telemetry toggles
  - `TELEMETRY_ENABLED`
  - `TELEMETRY_CONSOLE_ENABLED`
  - `TELEMETRY_OTLP_ENABLED`
  - `TELEMETRY_SERVICE_NAME`
- Bridge settings
  - `BRIDGE_ENABLED`
  - `BRIDGE_CORE_URL`
  - `BRIDGE_AI_URL`
  - `BRIDGE_TIMEOUT_MS`

Boolean config accepts: `1/0`, `true/false`, `yes/no`, `on/off`.

## Precedence and environment naming

Configuration precedence is intentionally simple and deterministic:

1. Explicit process environment at runtime (`process.env`)
2. Defaults defined in `@versa/config` schema

Environment variable naming conventions:

- Use upper snake case only.
- Use subsystem prefixes for clarity (`FEATURE_*`, `MCP_*`, `TELEMETRY_*`, `BRIDGE_*`).
- New cross-service settings must be added to `@versa/config` first, then consumed from that package.
- Avoid reading ad-hoc env keys directly from app/package code when the key represents shared runtime behavior.

## Dependency governance (apps vs packages)

- Apps (`apps/*`) may depend on workspace packages (`packages/*`) and external libraries.
- Packages (`packages/*`) may depend on other packages only when dependency direction preserves layering.
- Shared, foundational packages (`@versa/config`, `@versa/shared`) should remain low-level dependencies and should not import app code.
- App-to-app imports are forbidden.

## Import boundary notes

- `@versa/config`
  - Owns environment parsing and defaults.
  - Must not import app runtime modules.
  - Can be imported by any app/package that needs validated runtime settings.
- `@versa/shared`
  - Owns shared schemas/contracts and neutral types.
  - Must stay domain-contract focused; no app boot logic.
  - Can be imported by apps and other packages for canonical contracts.

These boundaries keep later workstreams additive and prevent config/schema drift across services.
