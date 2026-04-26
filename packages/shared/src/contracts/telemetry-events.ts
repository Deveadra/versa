import { z } from 'zod';

export const TelemetryEventNameEnum = z.enum([
  'system.startup',
  'system.shutdown',
  'api.request',
  'api.error',
  'task.created',
  'task.completed',
  'approval.requested',
  'approval.decided',
  'mcp.capability.invoked',
]);

export const TelemetryLevelEnum = z.enum(['debug', 'info', 'warn', 'error']);

export const TelemetryEventSchema = z.object({
  telemetryId: z.string().min(8),
  name: TelemetryEventNameEnum,
  level: TelemetryLevelEnum,
  timestamp: z.string().datetime(),
  actor: z.string().min(1),
  traceId: z.string().min(1),
  attributes: z.record(z.unknown()).default({}),
});

export type TelemetryEvent = z.infer<typeof TelemetryEventSchema>;

