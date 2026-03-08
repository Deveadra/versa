import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const defaultDbPath = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../data/versa.db');
const file = process.env.DATABASE_URL ?? defaultDbPath;
if (fs.existsSync(file)) {
  fs.rmSync(file);
}
console.log(`reset ${file}`);
