export type PlannerInputs = {
  overdueTasks: Array<{ title: string }>;
  todayTasks: Array<{ title: string }>;
  activeGoals: Array<{ title: string }>;
  todayBlocks: Array<{ title: string; status: string }>;
  studyPendingCount: number;
  followUpsSoon: number;
};

export const generateDailyPlan = (input: PlannerInputs) => {
  const priorities = [
    ...input.overdueTasks.slice(0, 2).map((task) => `Overdue: ${task.title}`),
    ...input.todayTasks.slice(0, 2).map((task) => `Today: ${task.title}`),
    ...(input.activeGoals[0] ? [`Goal: ${input.activeGoals[0].title}`] : []),
  ].slice(0, 3);

  return {
    generatedAt: new Date().toISOString(),
    priorities,
    overdueCount: input.overdueTasks.length,
    nextBlock: input.todayBlocks.find((block) => block.status === 'scheduled') ?? null,
    focusNext: priorities[0] ?? 'Plan your first task',
    studyPressure: input.studyPendingCount,
    followUpsSoon: input.followUpsSoon,
  };
};
