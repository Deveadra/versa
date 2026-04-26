import { z } from 'zod';

export const SkillExecutionModeEnum = z.enum(['sync', 'async']);

export const SkillDescriptorSchema = z.object({
  skillId: z.string().min(3),
  name: z.string().min(1),
  version: z.string().min(1),
  executionMode: SkillExecutionModeEnum,
  requiresApproval: z.boolean().default(false),
  inputSchemaRef: z.string().min(1),
  outputSchemaRef: z.string().min(1),
  tags: z.array(z.string()).default([]),
});

export type SkillDescriptor = z.infer<typeof SkillDescriptorSchema>;

