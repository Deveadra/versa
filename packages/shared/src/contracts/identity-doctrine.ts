import { z } from 'zod';

export const DoctrineModeEnum = z.enum(['strict', 'balanced', 'exploratory']);
export const RiskLevelEnum = z.enum(['low', 'medium', 'high']);

export const IdentityProfileSchema = z.object({
  agentId: z.string().min(3),
  displayName: z.string().min(1),
  persona: z.string().min(1),
  doctrineVersion: z.string().min(1),
  capabilities: z.array(z.string()).default([]),
});

export const DoctrinePolicySchema = z.object({
  policyId: z.string().min(3),
  version: z.string().min(1),
  mode: DoctrineModeEnum,
  allowAutonomousActions: z.boolean().default(false),
  maxRisk: RiskLevelEnum.default('low'),
  requiresApprovalAbove: RiskLevelEnum.default('low'),
});

export type IdentityProfile = z.infer<typeof IdentityProfileSchema>;
export type DoctrinePolicy = z.infer<typeof DoctrinePolicySchema>;

