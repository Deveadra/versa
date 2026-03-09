import fs from 'node:fs';
import path from 'node:path';
import { connectDb } from './index';

const db = connectDb();
db.exec('CREATE TABLE IF NOT EXISTS migrations (id TEXT PRIMARY KEY, applied_at TEXT NOT NULL);');

const migrationsDir = path.resolve(process.cwd(), 'migrations');
for (const file of fs.readdirSync(migrationsDir).filter((f) => f.endsWith('.sql')).sort()) {
  const exists = db.prepare('SELECT id FROM migrations WHERE id = ?').get(file);
  if (exists) continue;
  const sql = fs.readFileSync(path.join(migrationsDir, file), 'utf8');
  db.exec(sql);
  db.prepare('INSERT INTO migrations (id, applied_at) VALUES (?, ?)').run(file, new Date().toISOString());
}
console.log('migrations complete');
