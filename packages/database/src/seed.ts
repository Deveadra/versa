import { connectDb, goalRepo, jobRepo, scheduleRepo, studyRepo, taskRepo } from './index';

const db = connectDb();
const tasks = taskRepo(db);
const goals = goalRepo(db);
const schedules = scheduleRepo(db);
const study = studyRepo(db);
const jobs = jobRepo(db);

console.log('seed diagnostics: taskRepo methods', Object.keys(tasks));

if ((tasks.list() ?? []).length === 0) {
  const goal = goals.create({ title: 'Ship Phase 1 brainstem', whyItMatters: 'Daily reliability' });
  const task = tasks.create({ title: 'Review priorities', dueDate: new Date(Date.now() + 3600000).toISOString(), linkedGoalId: String(goal.id) });
  schedules.create({ title: 'Morning focus block', date: new Date().toISOString().slice(0, 10), startTime: '09:00', endTime: '10:00', linkedTaskId: String(task.id), linkedGoalId: String(goal.id) });
  const course = study.createCourse('Algorithms');
  study.createAssignment(String(course.id), 'Homework 1', new Date(Date.now() + 86400000).toISOString());
  const lead = jobs.createLead('Example Inc', 'Backend Engineer');
  jobs.convertLeadToApplication(String(lead.id));
}

console.log('seed complete');
