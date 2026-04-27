import { describe, expect, it } from 'vitest';
import { generateDailyPlan } from './planner';

describe('generateDailyPlan', () => {
  it('prioritizes overdue and today tasks before goals', () => {
    const plan = generateDailyPlan({
      overdueTasks: [{ title: 'Late task' }],
      todayTasks: [{ title: 'Today task' }],
      activeGoals: [{ title: 'Long-term goal' }],
      todayBlocks: [{ title: 'Focus', status: 'scheduled' }],
      studyPendingCount: 2,
      followUpsSoon: 1,
    });

    expect(plan.priorities[0]).toContain('Overdue');
    expect(plan.priorities[1]).toContain('Today');
    expect(plan.nextBlock?.title).toBe('Focus');
  });
});
