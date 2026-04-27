import { z } from 'zod';

const booleanFromEnv = z.preprocess((value) => {
  if (typeof value === 'boolean') return value;
  if (typeof value !== 'string') return value;

  const normalized = value.trim().toLowerCase();
  if (['1', 'true', 'yes', 'on'].includes(normalized)) return true;
  if (['0', 'false', 'no', 'off'].includes(normalized)) return false;

  return value;
}, z.boolean());

const RuntimeModeSchema = z.enum(['local', 'hybrid', 'cloud']);
const NodeEnvSchema = z.enum(['development', 'test', 'production']);

const ConfigSchema = z.object({
  CORE_PORT: z.coerce.number().default(4000),
  AI_PORT: z.coerce.number().default(4010),
  WEB_PORT: z.coerce.number().default(3000),
  MCP_GATEWAY_PORT: z.coerce.number().default(4020),

  NODE_ENV: NodeEnvSchema.default('development'),
  RUNTIME_MODE: RuntimeModeSchema.default('local'),

  DATABASE_URL: z.string().default('/workspace/versa/packages/database/data/versa.db'),
  DATABASE_READ_URL: z.string().default('/workspace/versa/packages/database/data/versa.db'),
  DATABASE_MIGRATIONS_PATH: z.string().default('/workspace/versa/packages/database/src/migrations'),

  FEATURE_MEMORY_ENABLED: booleanFromEnv.default(false),
  FEATURE_APPROVALS_ENABLED: booleanFromEnv.default(false),
  FEATURE_SKILLS_ENABLED: booleanFromEnv.default(false),
  FEATURE_WORKSPACES_ENABLED: booleanFromEnv.default(false),
  FEATURE_DOCTRINE_ENABLED: booleanFromEnv.default(true),
  DOCTRINE_PATH: z.string().default('state/doctrine.json'),

  MCP_ENABLED: booleanFromEnv.default(false),
  MCP_TRANSPORT: z.enum(['stdio', 'http']).default('stdio'),
  MCP_HOST: z.string().default('127.0.0.1'),
  MCP_PORT: z.coerce.number().default(4021),

  TELEMETRY_ENABLED: booleanFromEnv.default(false),
  TELEMETRY_CONSOLE_ENABLED: booleanFromEnv.default(true),
  TELEMETRY_OTLP_ENABLED: booleanFromEnv.default(false),
  TELEMETRY_SERVICE_NAME: z.string().default('versa'),

  BRIDGE_ENABLED: booleanFromEnv.default(false),
  BRIDGE_CORE_URL: z.string().url().default('http://127.0.0.1:4000'),
  BRIDGE_AI_URL: z.string().url().default('http://127.0.0.1:4010'),
  BRIDGE_TIMEOUT_MS: z.coerce.number().int().positive().default(4000),
});

export type VersaConfig = z.infer<typeof ConfigSchema>;

export const parseConfig = (env: Record<string, unknown> = process.env): VersaConfig =>
  ConfigSchema.parse(env);

export const loadConfig = (): VersaConfig => parseConfig(process.env);
