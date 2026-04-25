import { z } from 'zod';

export const TrustLevelEnum = z.enum(['readonly', 'assisted', 'supervised', 'autonomous']);
export const ApprovalDecisionEnum = z.enum(['approved', 'denied', 'expired']);

export const ApprovalRequestSchema = z.object({
  requestId: z.string().min(8),
  actionType: z.string().min(1),
  rationale: z.string().min(1),
  risk: z.enum(['low', 'medium', 'high']),
  requestedBy: z.string().min(1),
  requestedAt: z.string().datetime(),
});

export const ApprovalDecisionSchema = z.object({
  requestId: z.string().min(8),
  decision: ApprovalDecisionEnum,
  decidedBy: z.string().min(1),
  decidedAt: z.string().datetime(),
  reason: z.string().optional(),
});

export type ApprovalRequest = z.infer<typeof ApprovalRequestSchema>;
export type ApprovalDecision = z.infer<typeof ApprovalDecisionSchema>;

