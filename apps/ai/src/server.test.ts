import { AddressInfo } from 'node:net';
import { afterEach, describe, expect, it } from 'vitest';
import { createAiApp } from './server';

const servers: Array<{ close: (cb: () => void) => void }> = [];

const startTestServer = async () => {
  const { app } = createAiApp();
  const server = app.listen(0);
  servers.push(server);

  await new Promise<void>((resolve) => server.once('listening', () => resolve()));
  const port = (server.address() as AddressInfo).port;

  return {
    async request(path: string, init?: RequestInit) {
      const response = await fetch(`http://127.0.0.1:${port}${path}`, init);
      const json = await response.json();
      return { response, json };
    },
  };
};

afterEach(async () => {
  await Promise.all(
    servers.splice(0).map(
      (server) =>
        new Promise<void>((resolve) => {
          server.close(() => resolve());
        }),
    ),
  );

  delete process.env.FEATURE_APPROVALS_ENABLED;
  delete process.env.BRIDGE_ENABLED;
  delete process.env.BRIDGE_MODE;
  delete process.env.TELEMETRY_ENABLED;
});

describe('apps/ai runtime boundary', () => {
  it('returns health and advertised capabilities', async () => {
    const svc = await startTestServer();

    const { response: healthResponse, json: healthJson } = await svc.request('/health');
    expect(healthResponse.status).toBe(200);
    expect(healthJson).toEqual({ ok: true });

    const { response: capabilitiesResponse, json: capabilitiesJson } = await svc.request('/capabilities');
    expect(capabilitiesResponse.status).toBe(200);
    expect(capabilitiesJson.capabilities).toContain('search_memory');
    expect(capabilitiesJson.capabilities).toContain('generate_study_plan');
  });

  it('exposes bridge health and capabilities contracts', async () => {
    process.env.BRIDGE_ENABLED = 'true';
    process.env.BRIDGE_MODE = 'primary';

    const svc = await startTestServer();

    const { response: bridgeHealthResponse, json: bridgeHealthJson } = await svc.request('/bridge/health');
    expect(bridgeHealthResponse.status).toBe(200);
    expect(bridgeHealthJson.data.status).toBe('ok');
    expect(bridgeHealthJson.data.mode).toBe('primary');

    const { response: bridgeCapabilitiesResponse, json: bridgeCapabilitiesJson } =
      await svc.request('/bridge/capabilities');
    expect(bridgeCapabilitiesResponse.status).toBe(200);
    expect(bridgeCapabilitiesJson.data.some((cap: { id: string }) => cap.id === 'legacy.summarize_day')).toBe(
      true,
    );
  });

  it('maps bridge invoke disabled mode to 503 unavailable', async () => {
    process.env.BRIDGE_ENABLED = 'false';
    process.env.BRIDGE_MODE = 'disabled';

    const svc = await startTestServer();

    const { response, json } = await svc.request('/bridge/invoke', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ capabilityId: 'legacy.summarize_day', payload: {} }),
    });

    expect(response.status).toBe(503);
    expect(json.data.attempted).toBe(false);
    expect(json.data.mode).toBe('disabled');
  });

  it('enforces approvals for risky bridge invocation when enabled', async () => {
    process.env.FEATURE_APPROVALS_ENABLED = 'true';
    process.env.BRIDGE_ENABLED = 'true';
    process.env.BRIDGE_MODE = 'primary';

    const svc = await startTestServer();

    const { response, json } = await svc.request('/bridge/invoke', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ capabilityId: 'legacy.rank_priorities', payload: { risky: true } }),
    });

    expect(response.status).toBe(409);
    expect(json.error).toContain('requires operator approval');
    expect(json.approval.outcome).toBe('require_approval');
  });

  it('routes legacy ai/execute requests through bridge with status mapping', async () => {
    process.env.BRIDGE_ENABLED = 'false';
    process.env.BRIDGE_MODE = 'disabled';

    const svc = await startTestServer();

    const { response, json } = await svc.request('/ai/execute', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ capabilityId: 'legacy.summarize_day', input: {} }),
    });

    expect(response.status).toBe(503);
    expect(json.data.target).toBe('legacy_python_runtime');
    expect(json.data.status).toBe('unavailable');
  });

  it('maps missing skills on ai/execute to failed contract response', async () => {
    const svc = await startTestServer();

    const { response, json } = await svc.request('/ai/execute', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ capabilityId: 'missing-skill', input: {} }),
    });

    expect(response.status).toBe(422);
    expect(json.data.target).toBe('typescript_service');
    expect(json.data.status).toBe('failed');
    expect(json.data.error.code).toBe('SKILL_NOT_FOUND');
  });

  it('returns 404 for missing skill execution endpoint requests', async () => {
    const svc = await startTestServer();

    const { response, json } = await svc.request('/skills/execute', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ skillId: 'missing-skill', input: {} }),
    });

    expect(response.status).toBe(404);
    expect(json.data.error.code).toBe('SKILL_NOT_FOUND');
  });

  it('requires approval for medium-risk skill execution below safe-act trust', async () => {
    process.env.FEATURE_APPROVALS_ENABLED = 'true';

    const svc = await startTestServer();

    const { response, json } = await svc.request('/skills/execute', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        skillId: 'repo-inspection',
        input: { files: ['README.md'] },
        context: { trustLevel: 'draft' },
      }),
    });

    expect(response.status).toBe(409);
    expect(json.error).toContain('requires operator approval');
    expect(json.approval.outcome).toBe('require_approval');
  });
});
