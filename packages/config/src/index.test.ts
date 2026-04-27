import { describe, expect, it } from 'vitest';
import { parseConfig } from './index';

describe('parseConfig', () => {
  it('applies defaults for canonical config surface', () => {
    const parsed = parseConfig({});

    expect(parsed.CORE_PORT).toBe(4000);
    expect(parsed.AI_PORT).toBe(4010);
    expect(parsed.WEB_PORT).toBe(3000);
    expect(parsed.MCP_GATEWAY_PORT).toBe(4020);

    expect(parsed.RUNTIME_MODE).toBe('local');
    expect(parsed.NODE_ENV).toBe('development');

    expect(parsed.FEATURE_MEMORY_ENABLED).toBe(false);
    expect(parsed.FEATURE_APPROVALS_ENABLED).toBe(false);
    expect(parsed.FEATURE_SKILLS_ENABLED).toBe(false);
    expect(parsed.FEATURE_WORKSPACES_ENABLED).toBe(false);
    expect(parsed.FEATURE_DOCTRINE_ENABLED).toBe(true);
    expect(parsed.DOCTRINE_PATH).toBe('state/doctrine.json');

    expect(parsed.MCP_ENABLED).toBe(false);
    expect(parsed.TELEMETRY_ENABLED).toBe(false);
    expect(parsed.BRIDGE_ENABLED).toBe(false);
    expect(parsed.BRIDGE_MODE).toBe('disabled');
    expect(parsed.BRIDGE_LEGACY_RUNTIME_URL).toBe('http://127.0.0.1:8000');
    expect(parsed.BRIDGE_HEALTH_PATH).toBe('/health');
    expect(parsed.BRIDGE_CAPABILITIES_PATH).toBe('/capabilities');
    expect(parsed.BRIDGE_INVOKE_PATH).toBe('/invoke');
  });

  it('coerces booleans, numbers, and validates enums', () => {
    const parsed = parseConfig({
      CORE_PORT: '4100',
      NODE_ENV: 'production',
      RUNTIME_MODE: 'hybrid',
      FEATURE_MEMORY_ENABLED: 'true',
      FEATURE_DOCTRINE_ENABLED: '0',
      MCP_ENABLED: '1',
      TELEMETRY_OTLP_ENABLED: 'yes',
      BRIDGE_ENABLED: 'on',
      BRIDGE_MODE: 'shadow',
      BRIDGE_LEGACY_RUNTIME_URL: 'http://127.0.0.1:18000',
      BRIDGE_TIMEOUT_MS: '9000',
      BRIDGE_HEALTH_PATH: '/runtime/health',
      BRIDGE_CAPABILITIES_PATH: '/runtime/capabilities',
      BRIDGE_INVOKE_PATH: '/runtime/invoke',
      MCP_TRANSPORT: 'http',
    });

    expect(parsed.CORE_PORT).toBe(4100);
    expect(parsed.NODE_ENV).toBe('production');
    expect(parsed.RUNTIME_MODE).toBe('hybrid');
    expect(parsed.FEATURE_MEMORY_ENABLED).toBe(true);
    expect(parsed.FEATURE_DOCTRINE_ENABLED).toBe(false);
    expect(parsed.MCP_ENABLED).toBe(true);
    expect(parsed.TELEMETRY_OTLP_ENABLED).toBe(true);
    expect(parsed.BRIDGE_ENABLED).toBe(true);
    expect(parsed.BRIDGE_MODE).toBe('shadow');
    expect(parsed.BRIDGE_LEGACY_RUNTIME_URL).toBe('http://127.0.0.1:18000');
    expect(parsed.BRIDGE_TIMEOUT_MS).toBe(9000);
    expect(parsed.BRIDGE_HEALTH_PATH).toBe('/runtime/health');
    expect(parsed.BRIDGE_CAPABILITIES_PATH).toBe('/runtime/capabilities');
    expect(parsed.BRIDGE_INVOKE_PATH).toBe('/runtime/invoke');
    expect(parsed.MCP_TRANSPORT).toBe('http');
  });

  it('rejects invalid values', () => {
    expect(() => parseConfig({ RUNTIME_MODE: 'legacy' })).toThrow();
    expect(() => parseConfig({ BRIDGE_MODE: 'legacy' })).toThrow();
    expect(() => parseConfig({ BRIDGE_CORE_URL: 'not-a-url' })).toThrow();
    expect(() => parseConfig({ BRIDGE_LEGACY_RUNTIME_URL: 'not-a-url' })).toThrow();
    expect(() => parseConfig({ FEATURE_MEMORY_ENABLED: 'definitely' })).toThrow();
  });
});
