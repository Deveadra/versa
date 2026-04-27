import {
  BridgeCapabilitySchema,
  BridgeHealthStatusSchema,
  BridgeAdapterResultSchema,
  LegacyBridgeRequestSchema,
  LegacyBridgeResponseSchema,
  type BridgeCapability,
  type BridgeExecutionMode,
  type BridgeHealthStatus,
  type BridgeAdapterResult,
  type LegacyBridgeRequest,
  type LegacyBridgeResponse,
} from '@versa/shared';

export type BridgeConfig = {
  BRIDGE_ENABLED: boolean;
  BRIDGE_MODE: BridgeExecutionMode;
  BRIDGE_LEGACY_RUNTIME_URL: string;
  BRIDGE_HEALTH_PATH: string;
  BRIDGE_CAPABILITIES_PATH: string;
  BRIDGE_INVOKE_PATH: string;
};

type LegacyCapabilitySeed = Omit<BridgeCapability, 'owner' | 'status' | 'metadata'> & {
  metadata?: Record<string, unknown>;
};

const defaultCapabilities: LegacyCapabilitySeed[] = [
  {
    id: 'legacy.summarize_day',
    name: 'summarize_day',
    description: 'Legacy runtime daily summarization capability',
    version: 'legacy-v1',
  },
  {
    id: 'legacy.generate_study_plan',
    name: 'generate_study_plan',
    description: 'Legacy runtime study-plan generation capability',
    version: 'legacy-v1',
  },
  {
    id: 'legacy.rank_priorities',
    name: 'rank_priorities',
    description: 'Legacy runtime priority ranking capability',
    version: 'legacy-v1',
  },
];

const normalizePath = (path: string) => (path.startsWith('/') ? path : `/${path}`);

export const buildLegacyBridgeEndpoint = (baseUrl: string, path: string): string => {
  const normalizedBase = baseUrl.endsWith('/') ? baseUrl.slice(0, -1) : baseUrl;
  return `${normalizedBase}${normalizePath(path)}`;
};

export const getDefaultLegacyCapabilities = (): BridgeCapability[] =>
  defaultCapabilities.map((capability) =>
    BridgeCapabilitySchema.parse({
      ...capability,
      owner: 'legacy_python_runtime',
      status: 'available',
      metadata: capability.metadata ?? {},
    }),
  );

export const buildBridgeHealth = (
  config: BridgeConfig,
  input?: {
    latencyMs?: number;
    status?: BridgeHealthStatus['status'];
    details?: Record<string, unknown>;
  },
): BridgeHealthStatus =>
  BridgeHealthStatusSchema.parse({
    service: 'legacy-python-bridge',
    status: input?.status ?? (config.BRIDGE_ENABLED ? 'ok' : 'degraded'),
    mode: config.BRIDGE_MODE,
    targetRuntime: 'legacy_python',
    endpoint: buildLegacyBridgeEndpoint(config.BRIDGE_LEGACY_RUNTIME_URL, config.BRIDGE_HEALTH_PATH),
    latencyMs: input?.latencyMs,
    lastCheckedAt: new Date().toISOString(),
    details: input?.details ?? {},
  });

export const createBridgeAdapter = (config: BridgeConfig) => {
  const health = () => buildBridgeHealth(config);

  const capabilities = () => getDefaultLegacyCapabilities();

  const execute = (request: LegacyBridgeRequest): BridgeAdapterResult => {
    const parsed = LegacyBridgeRequestSchema.parse(request);

    if (!config.BRIDGE_ENABLED || config.BRIDGE_MODE === 'disabled') {
      return BridgeAdapterResultSchema.parse({
        bridgeEnabled: false,
        mode: config.BRIDGE_MODE,
        attempted: false,
        fallbackTarget: 'typescript_service',
      });
    }

    const response = LegacyBridgeResponseSchema.parse({
      requestId: parsed.requestId,
      operation: parsed.operation,
      status: 'ok',
      targetRuntime: 'legacy_python',
      capabilityId: parsed.capabilityId,
      data:
        parsed.operation === 'health'
          ? {
              health: health(),
            }
          : parsed.operation === 'capabilities'
            ? {
                capabilities: capabilities(),
              }
            : {
                proxied: true,
                mode: config.BRIDGE_MODE,
                endpoint: buildLegacyBridgeEndpoint(config.BRIDGE_LEGACY_RUNTIME_URL, config.BRIDGE_INVOKE_PATH),
                payload: parsed.payload,
              },
    });

    return BridgeAdapterResultSchema.parse({
      bridgeEnabled: true,
      mode: config.BRIDGE_MODE,
      attempted: true,
      response,
      fallbackTarget: config.BRIDGE_MODE === 'shadow' ? 'typescript_service' : undefined,
    });
  };

  return {
    health,
    capabilities,
    execute,
  };
};

export type { BridgeCapability, BridgeHealthStatus, LegacyBridgeRequest, LegacyBridgeResponse };
