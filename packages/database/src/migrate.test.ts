import { describe, expect, it } from 'vitest';
import { execSync } from 'node:child_process';
import { connectDb } from './index';

describe('migration smoke', () => {
  it('creates tasks table', () => {
    process.env.DATABASE_URL = 'data/test.db';
    execSync('pnpm --filter @versa/database reset', { stdio: 'pipe' });
    execSync('pnpm --filter @versa/database migrate', { stdio: 'pipe' });
    const db = connectDb(process.env.DATABASE_URL);
    const row = db.prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='tasks'").get() as
      | { name: string }
      | undefined;
    expect(row?.name).toBe('tasks');
  });
});
