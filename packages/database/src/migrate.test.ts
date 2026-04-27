import { execSync } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { connectDb } from './index';

const packageRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const repoRoot = resolve(packageRoot, '../..');
const testDatabaseUrl = resolve(packageRoot, 'data/test.db');

const logNativeModuleDiagnostics = (env: NodeJS.ProcessEnv) => {
  const runtime = execSync("node -p \"process.version + ' modules=' + process.versions.modules\"", {
    cwd: repoRoot,
    env,
    stdio: 'pipe',
  })
    .toString()
    .trim();

  const nativeBinaryPath = execSync(
    "node -e \"const path=require('node:path'); const pkg=require.resolve('better-sqlite3/package.json'); const dir=path.dirname(pkg); console.log(path.join(dir, 'build/Release/better_sqlite3.node'));\"",
    {
      cwd: repoRoot,
      env,
      stdio: 'pipe',
    },
  )
    .toString()
    .trim();

  console.info('[migration smoke] Node runtime:', runtime);
  console.info('[migration smoke] better-sqlite3 binary:', nativeBinaryPath);
};

const ensureBetterSqlite3ForActiveNode = (env: NodeJS.ProcessEnv) => {
  const verifyLoad = () => {
    execSync("node -e \"require('better-sqlite3');\"", {
      cwd: repoRoot,
      env,
      stdio: 'pipe',
    });
  };

  try {
    verifyLoad();
  } catch {
    execSync('pnpm --filter @versa/database rebuild better-sqlite3', {
      cwd: repoRoot,
      env,
      stdio: 'pipe',
    });

    verifyLoad();
  }
};

describe('migration smoke', () => {
  it('creates required tables', () => {
    const env = {
      ...process.env,
      DATABASE_URL: testDatabaseUrl,
      npm_config_nodedir: '/usr',
    };

    logNativeModuleDiagnostics(env);
    ensureBetterSqlite3ForActiveNode(env);

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

    const tasksRow = db
      .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='tasks'")
      .get() as { name: string } | undefined;
    expect(tasksRow?.name).toBe('tasks');

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

    const environmentRow = db
      .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='environments'")
      .get() as { name: string } | undefined;
    expect(environmentRow?.name).toBe('environments');

    const environmentRecordRow = db
      .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='environment_records'")
      .get() as { name: string } | undefined;
    expect(environmentRecordRow?.name).toBe('environment_records');

    const environmentRelationshipRow = db
      .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='environment_relationships'")
      .get() as { name: string } | undefined;
    expect(environmentRelationshipRow?.name).toBe('environment_relationships');

    const environmentAccessPathRow = db
      .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='environment_access_paths'")
      .get() as { name: string } | undefined;
    expect(environmentAccessPathRow?.name).toBe('environment_access_paths');

    const environmentProcedureRow = db
      .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='environment_procedures'")
      .get() as { name: string } | undefined;
    expect(environmentProcedureRow?.name).toBe('environment_procedures');

    db.close();
  }, 30_000);
});
