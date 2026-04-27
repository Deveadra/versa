import {
  CapabilityLookupResultSchema,
  CapabilityRegistrationResultSchema,
  CapabilityRegistryEntrySchema,
  GatewayHealthStatusSchema,
  type CapabilityMetadata,
  type CapabilityLookupResult,
  type CapabilityRegistrationResult,
  type CapabilityRegistryEntry,
  type GatewayHealthStatus,
  type McpPromptDefinition,
  type McpResourceDefinition,
  type McpTransport,
  type McpToolDefinition,
} from '@versa/shared';

type GatewayRuntimeConfig = {
  MCP_ENABLED: boolean;
  MCP_TRANSPORT: McpTransport;
  TELEMETRY_ENABLED: boolean;
};

const makeMetadata = (id: string, title: string, summary: string): CapabilityMetadata => ({
  title,
  summary,
  owner: 'versa-platform',
  lifecycle: 'active',
  sensitivity: 'internal',
  tags: ['ws09', 'mcp', id],
  approvals: {
    required: true,
    policyRef: 'ws08.approvals.default',
    writeAllowed: false,
  },
});

const readMemoryResource: McpResourceDefinition = {
  id: 'resource.memory.read',
  name: 'memory.read',
  description: 'Read-only memory query surface routed through the canonical gateway.',
  uriTemplate: '/memory?q={text}&tier={tier}&limit={limit}',
  methods: ['GET'],
  transport: 'http',
};

const workspaceContextResource: McpResourceDefinition = {
  id: 'resource.workspace.context',
  name: 'workspace.context',
  description: 'Read-only workspace context bundle access.',
  uriTemplate: '/workspaces/{slug}/context?limit={limit}',
  methods: ['GET'],
  transport: 'http',
};

const environmentContextResource: McpResourceDefinition = {
  id: 'resource.environment.context',
  name: 'environment.context',
  description: 'Read-only environment twin context bundle access.',
  uriTemplate: '/environments/{slug}/context?limit={limit}',
  methods: ['GET'],
  transport: 'http',
};

const listSkillsTool: McpToolDefinition = {
  id: 'tool.skills.list',
  name: 'skills.list',
  description: 'List internal foundational skills available for controlled execution.',
  inputSchema: {
    type: 'object',
    properties: {},
  },
  outputSchema: {
    type: 'object',
    properties: {
      skills: {
        type: 'array',
      },
    },
  },
  sideEffectLevel: 'read',
  approvalsRequired: true,
};

const readDailySummaryPrompt: McpPromptDefinition = {
  id: 'prompt.daily.summary',
  name: 'daily.summary',
  description: 'Prompt/workflow definition for daily context summarization.',
  inputSchema: {
    type: 'object',
    properties: {
      date: {
        type: 'string',
      },
    },
  },
  outputSchema: {
    type: 'object',
    properties: {
      summary: {
        type: 'string',
      },
    },
  },
  requiresApproval: true,
};

export const foundationalRegistryEntries: CapabilityRegistryEntry[] = [
  CapabilityRegistryEntrySchema.parse({
    capabilityId: 'cap.memory.read',
    kind: 'resource',
    metadata: makeMetadata('memory-read', 'Memory read', 'Read memory through governed MCP boundary.'),
    resources: [readMemoryResource],
    tools: [],
    prompts: [],
    status: 'active',
  }),
  CapabilityRegistryEntrySchema.parse({
    capabilityId: 'cap.workspace.context',
    kind: 'resource',
    metadata: makeMetadata(
      'workspace-context',
      'Workspace context read',
      'Read workspace context bundles through gateway registry.',
    ),
    resources: [workspaceContextResource],
    tools: [],
    prompts: [],
    status: 'active',
  }),
  CapabilityRegistryEntrySchema.parse({
    capabilityId: 'cap.environment.context',
    kind: 'resource',
    metadata: makeMetadata(
      'environment-context',
      'Environment context read',
      'Read environment twin context bundles through gateway registry.',
    ),
    resources: [environmentContextResource],
    tools: [],
    prompts: [],
    status: 'active',
  }),
  CapabilityRegistryEntrySchema.parse({
    capabilityId: 'cap.skills.list',
    kind: 'tool',
    metadata: makeMetadata('skills-list', 'Skills list', 'Enumerate skill metadata through gateway tools.'),
    resources: [],
    tools: [listSkillsTool],
    prompts: [],
    status: 'active',
  }),
  CapabilityRegistryEntrySchema.parse({
    capabilityId: 'cap.prompt.daily-summary',
    kind: 'prompt',
    metadata: makeMetadata(
      'daily-summary',
      'Daily summary prompt',
      'Prompt/workflow contract for daily summary generation.',
    ),
    resources: [],
    tools: [],
    prompts: [readDailySummaryPrompt],
    status: 'active',
  }),
];

const mapById = new Map<string, CapabilityRegistryEntry>(
  foundationalRegistryEntries.map((entry) => [entry.capabilityId, entry]),
);

export const listCapabilities = (): CapabilityRegistrationResult =>
  CapabilityRegistrationResultSchema.parse({
    registered: foundationalRegistryEntries,
    count: foundationalRegistryEntries.length,
    status: 'ok',
  });

export const lookupCapability = (capabilityId: string): CapabilityLookupResult =>
  CapabilityLookupResultSchema.parse({
    capabilityId,
    found: mapById.has(capabilityId),
    entry: mapById.get(capabilityId),
  });

export const buildGatewayHealth = (cfg: GatewayRuntimeConfig, uptimeMs: number): GatewayHealthStatus =>
  GatewayHealthStatusSchema.parse({
    service: 'mcp-gateway',
    status: cfg.MCP_ENABLED ? 'ok' : 'degraded',
    transport: cfg.MCP_TRANSPORT,
    uptimeMs: Math.max(0, Math.round(uptimeMs)),
    registeredCapabilities: foundationalRegistryEntries.length,
    telemetryEnabled: cfg.TELEMETRY_ENABLED,
    approvalsRequiredByDefault: foundationalRegistryEntries.every(
      (entry) => entry.metadata.approvals.required,
    ),
    timestamp: new Date().toISOString(),
  });
