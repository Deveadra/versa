# WS09 — MCP gateway and capability mesh

Issue: `https://github.com/Deveadra/versa/issues/54`

## Objective

Establish a canonical MCP gateway app and shared capability-registry contracts so MCP-facing integrations route through one governed edge surface instead of being scattered across unrelated services.

## Canonical app decision

WS09 establishes one canonical MCP gateway app: `@versa/mcp-gateway` in [`apps/mcp-gateway`](../../apps/mcp-gateway).

- `@versa/mcp-gateway` owns MCP-facing capability registration and lookup surfaces.
- Foundational capability exposure in this slice is read-first and approval-aware by default.

## Shared contract surface

WS09 adds MCP gateway and capability-mesh contracts in [`packages/shared/src/index.ts`](../../packages/shared/src/index.ts):

- `McpTransportEnum`
- `CapabilityKindEnum`
- `McpResourceDefinitionSchema`
- `McpToolDefinitionSchema`
- `McpPromptDefinitionSchema`
- `CapabilityApprovalMetadataSchema`
- `CapabilityMetadataSchema`
- `CapabilityRegistryEntrySchema`
- `GatewayHealthStatusSchema`
- `CapabilityRegistrationResultSchema`
- `CapabilityLookupResultSchema`

These contracts make registry entries, health/status, and lookup behavior explicit and importable for downstream workstreams.

## Foundational gateway behavior

The gateway app in [`apps/mcp-gateway/src/server.ts`](../../apps/mcp-gateway/src/server.ts) exposes additive, bounded operational surfaces:

- `GET /health`
- `GET /mcp/health`
- `GET /mcp/capabilities`
- `GET /mcp/capabilities/:capabilityId`

Registry foundations are implemented in [`apps/mcp-gateway/src/registry.ts`](../../apps/mcp-gateway/src/registry.ts), including:

- foundational capability entries for memory/workspace/environment read surfaces
- a foundational tool contract example (`skills.list`)
- a foundational prompt/workflow contract example (`daily.summary`)

## Config and telemetry linkage

Gateway startup uses canonical config from [`packages/config/src/index.ts`](../../packages/config/src/index.ts):

- `MCP_GATEWAY_PORT`
- `MCP_ENABLED`
- `MCP_TRANSPORT`
- telemetry toggles (`TELEMETRY_*`)

Gateway request and lifecycle telemetry uses the existing logging foundation from `@versa/logging`, including request middleware and startup/shutdown events.

## Boundary guidance

- MCP wrappers and registry logic belong in `apps/mcp-gateway`, not in unrelated app layers.
- This slice does not enable unrestricted write tooling.
- Approval-aware defaults are explicit in capability metadata (`approvals.required: true`, `writeAllowed: false`).
- Full external MCP rollout across every subsystem remains out of scope for WS09.

## Validation footprint

Foundational gateway/registry tests are added in [`apps/mcp-gateway/src/registry.test.ts`](../../apps/mcp-gateway/src/registry.test.ts), with shared-contract coverage extended in [`packages/shared/src/index.test.ts`](../../packages/shared/src/index.test.ts).
