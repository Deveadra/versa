import { z } from 'zod';

export const EnvironmentClassEnum = z.enum(['local', 'staging', 'production']);

export const EnvironmentTwinSchema = z.object({
  environmentId: z.string().min(3),
  class: EnvironmentClassEnum,
  os: z.string().min(1),
  shell: z.string().min(1),
  workspaceRoot: z.string().min(1),
  activeTerminals: z.array(z.string()).default([]),
  observedAt: z.string().datetime(),
});

export type EnvironmentTwin = z.infer<typeof EnvironmentTwinSchema>;

