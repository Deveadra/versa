import { z } from 'zod';

export const McpTransportEnum = z.enum(['stdio', 'http', 'websocket']);

export const McpCapabilitySchema = z.object({
  capabilityId: z.string().min(3),
  name: z.string().min(1),
  version: z.string().min(1),
  description: z.string().min(1),
  inputSchemaRef: z.string().min(1),
  outputSchemaRef: z.string().min(1),
  requiresApproval: z.boolean().default(false),
});

export const McpServerRegistrationSchema = z.object({
  serverId: z.string().min(3),
  transport: McpTransportEnum,
  endpoint: z.string().min(1),
  capabilities: z.array(McpCapabilitySchema).default([]),
  registeredAt: z.string().datetime(),
});

export type McpCapability = z.infer<typeof McpCapabilitySchema>;
export type McpServerRegistration = z.infer<typeof McpServerRegistrationSchema>;

