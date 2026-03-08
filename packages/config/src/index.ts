import { z } from 'zod';

const ConfigSchema = z.object({
  CORE_PORT: z.coerce.number().default(4000),
  AI_PORT: z.coerce.number().default(4010),
  DATABASE_URL: z.string().default('/workspace/versa/packages/database/data/versa.db'),
});

export const loadConfig = () => ConfigSchema.parse(process.env);
