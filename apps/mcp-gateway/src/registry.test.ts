import { describe, expect, it } from 'vitest';
import { buildGatewayHealth, listCapabilities, lookupCapability } from './registry';

describe('mcp gateway foundational registry', () => {
  it('lists foundational capabilities through shared contracts', () => {
    const registration = listCapabilities();

    expect(registration.status).toBe('ok');
    expect(registration.count).toBe(5);
  });

  it('looks up known capability and returns not-found for unknown capability', () => {
    const found = lookupCapability('cap.workspace.context');
    const missing = lookupCapability('cap.unknown');

    expect(found.found).toBe(true);
    expect(found.entry?.capabilityId).toBe('cap.workspace.context');
    expect(missing.found).toBe(false);
    expect(missing.entry).toBeUndefined();
  });

  it('builds health with degraded status when MCP is disabled', () => {
    const health = buildGatewayHealth(
      {
        MCP_ENABLED: false,
        MCP_TRANSPORT: 'http',
        TELEMETRY_ENABLED: true,
      },
      1234,
    );

    expect(health.service).toBe('mcp-gateway');
    expect(health.status).toBe('degraded');
    expect(health.transport).toBe('http');
    expect(health.approvalsRequiredByDefault).toBe(true);
  });
});
