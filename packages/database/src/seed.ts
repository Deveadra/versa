import { connectDb, taskRepo } from './index';

const db = connectDb();
const repo = taskRepo(db);
if ((repo.listTasks() ?? []).length === 0) {
  repo.createTask({ title: 'Phase 0 seeded task', description: 'Validate e2e task flow' });
}
console.log('seed complete');
