import express, { Request, Response } from 'express';
import { loadConfig } from '@versa/config';

const app = express();
const cfg = loadConfig();

const capabilities = {
  summarize_day: () => ({ summary: 'placeholder summary' }),
  generate_study_plan: () => ({ plan: [] }),
  rank_priorities: () => ({ priorities: [] }),
  record_memory: () => ({ ok: true }),
  search_memory: () => ({ results: [] }),
};

app.get('/health', (_req: Request, res: Response) => res.json({ ok: true }));
app.get('/capabilities', (_req: Request, res: Response) => res.json({ capabilities: Object.keys(capabilities) }));

app.listen(cfg.AI_PORT, () => {
  console.log(`ai started on ${cfg.AI_PORT}`);
});
