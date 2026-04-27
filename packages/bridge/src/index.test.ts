import { describe, expect, it } from 'vitest';
import { LegacyBridgeRequestSchema } from '@versa/shared';
import {
  buildLegacyBridgeEndpoint,
  buildBridgeHealth,
  createBridgeAdapter,
  getDefaultLegacyCapabilities,
} from './index';

describe('bridge adapter foundations', () => {
  const config = {
    BRIDGE_ENABLED: true,
    BRIDGE_MODE: 'primary' as const,
    BRIDGE_LEGACY_RUNTIME_URL: 'http://127.0.0.1:8000',
    BRIDGE_HEALTH_PATH: '/health',
    BRIDGE_CAPABILITIES_PATH: '/capabilities',
    BRIDGE_INVOKE_PATH: '/invoke',
  };

  it('builds normalized endpoint paths', () => {
    expect(buildLegacyBridgeEndpoint('http://127.0.0.1:8000', '/health')).toBe(
      'http://127.0.0.1:8000/health',
    );
    expect(buildLegacyBridgeEndpoint('http://127.0.0.1:8000/', 'invoke')).toBe(
      'http://127.0.0.1:8000/invoke',
    );
  });

  it('exposes default legacy capabilities as typed entries', () => {
    const capabilities = getDefaultLegacyCapabilities();
    expect(capabilities.length).toBeGreaterThan(0);
    expect(capabilities[0]?.owner).toBe('legacy_python_runtime');
  });

  it('builds bridge health with bridge mode/runtime typing', () => {
    const health = buildBridgeHealth(config, {
      latencyMs: 15,
      status: 'ok',
    });

    expect(health.mode).toBe('primary');
    expect(health.targetRuntime).toBe('legacy_python');
    expect(health.status).toBe('ok');
  });

  it('executes invoke operation in primary mode and returns typed adapter result', () => {
    const adapter = createBridgeAdapter(config);
    const request = LegacyBridgeRequestSchema.parse({
      requestId: 'breq_12345678',
      operation: 'invoke',
      capabilityId: 'legacy.summarize_day',
      payload: { text: 'hello' },
      context: {
        traceId: 'trace-123',
      },
    });

    const result = adapter.execute(request);
    expect(result.bridgeEnabled).toBe(true);
    expect(result.attempted).toBe(true);
    expect(result.response?.status).toBe('ok');
  });

  it('returns fallback signal when bridge mode is disabled', () => {
    const adapter = createBridgeAdapter({
      ...config,
      BRIDGE_ENABLED: false,
      BRIDGE_MODE: 'disabled',
    });
    const request = LegacyBridgeRequestSchema.parse({
      requestId: 'breq_87654321',
      operation: 'health',
      payload: {},
      context: {},
    });

    const result = adapter.execute(request);
    expect(result.attempted).toBe(false);
    expect(result.fallbackTarget).toBe('typescript_service');
  });
});
