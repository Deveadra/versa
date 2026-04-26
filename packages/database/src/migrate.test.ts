import { execSync } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { connectDb } from './index';

const packageRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const repoRoot = resolve(packageRoot, '../..');
const testDatabaseUrl = resolve(packageRoot, 'data/test.db');

describe('migration smoke', () => {
  it('creates tasks table', () => {
    const env = {
      ...process.env,
      DATABASE_URL: testDatabaseUrl,
    };

    execSync('pnpm --filter @versa/database reset', {
      cwd: repoRoot,
      env,
      stdio: 'pipe',
    });

    execSync('pnpm --filter @versa/database migrate', {
      cwd: repoRoot,
      env,
      stdio: 'pipe',
    });

    const db = connectDb(testDatabaseUrl);

    const row = db
      .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='tasks'")
      .get() as { name: string } | undefined;

    expect(row?.name).toBe('tasks');

    const memoryRow = db
      .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='memories'")
      .get() as { name: string } | undefined;

    expect(memoryRow?.name).toBe('memories');

    const workspaceRow = db
      .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='workspaces'")
      .get() as { name: string } | undefined;

    expect(workspaceRow?.name).toBe('workspaces');

    const workspaceCheckpointRow = db
      .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='workspace_checkpoints'")
      .get() as { name: string } | undefined;

    expect(workspaceCheckpointRow?.name).toBe('workspace_checkpoints');

    db.close();
  });
});
