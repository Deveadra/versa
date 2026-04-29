import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

import {
  buildTaskCardFileName,
  buildTaskCardPath,
  createRooHandoffRenderModel,
  createTaskCardRenderModel,
  extractIssueRequirements,
  GitHubIssueIntakeService,
  normalizeGitHubIssue,
  prepareSandboxExecution,
  renderRooHandoffMarkdown,
  refreshTaskCardMarkdown,
  renderTaskCardMarkdown,
  type GitHubIssueReader,
} from './index';
import {
  EPIC_ISSUE_FIXTURE,
  MINIMAL_ISSUE_FIXTURE,
  WS14_TASK_CARD_ISSUE_FIXTURE,
  WS15_ROO_HANDOFF_ISSUE_FIXTURE,
  WORKSTREAM_ISSUE_FIXTURE,
} from './fixtures/github-issues';

describe('extractIssueRequirements', () => {
  it('extracts structured requirements from workstream issue body', () => {
    const parsed = extractIssueRequirements(WORKSTREAM_ISSUE_FIXTURE.body ?? '');

    expect(parsed.parentEpic).toBe(77);
    expect(parsed.objective).toContain('Create the GitHub issue intake layer');
    expect(parsed.why).toContain('source of authority');
    expect(parsed.deliverables).toContain('GitHub issue intake contract');
    expect(parsed.expectedCodeChanges).toContain('orchestrator-facing intake service');
    expect(parsed.acceptanceCriteria).toContain(
      'downstream workstreams can consume a stable IssueIntake object',
    );
    expect(parsed.constraints).toContain('do not generate task cards in this workstream');
    expect(parsed.suggestedBranch).toBe('orchestrator/ws13-github-issue-intake');
    expect(parsed.suggestedPrTitle).toBe('orchestrator(ws13): add GitHub issue intake foundation');
    expect(parsed.dependencies).toEqual([77]);
    expect(parsed.blockers).toEqual([79, 80]);
  });

  it('handles missing optional sections gracefully', () => {
    const parsed = extractIssueRequirements(EPIC_ISSUE_FIXTURE.body ?? '');

    expect(parsed.parentEpic).toBeNull();
    expect(parsed.suggestedBranch).toBeNull();
    expect(parsed.suggestedPrTitle).toBeNull();
    expect(parsed.dependencies).toEqual([]);
    expect(parsed.blockers).toEqual([]);
    expect(parsed.acceptanceCriteria.length).toBeGreaterThan(0);
  });

  it('does not consume following labels when labeled value is blank', () => {
    const parsed = extractIssueRequirements(`## Goal

Small objective

Suggested branch

Suggested PR title

Depends On: #77`);

    expect(parsed.suggestedBranch).toBeNull();
    expect(parsed.suggestedPrTitle).toBeNull();
    expect(parsed.dependencies).toEqual([77]);
  });
});

describe('normalizeGitHubIssue', () => {
  it('normalizes metadata and requirements into stable IssueIntake', () => {
    const intake = normalizeGitHubIssue(WORKSTREAM_ISSUE_FIXTURE);

    expect(intake.metadata.number).toBe(78);
    expect(intake.metadata.state).toBe('open');
    expect(intake.metadata.labels).toEqual(['orchestrator', 'ws13']);
    expect(intake.metadata.assignees).toEqual(['Deveadra']);
    expect(intake.requirements.parentEpic).toBe(77);
  });

  it('represents missing optional body fields without crashing', () => {
    const intake = normalizeGitHubIssue(MINIMAL_ISSUE_FIXTURE);

    expect(intake.metadata.body).toBe('');
    expect(intake.requirements.objective).toBeNull();
    expect(intake.requirements.why).toBeNull();
    expect(intake.requirements.deliverables).toEqual([]);
    expect(intake.requirements.acceptanceCriteria).toEqual([]);
    expect(intake.requirements.constraints).toEqual([]);
  });
});

describe('GitHubIssueIntakeService', () => {
  it('fetches issue by repo and issue number and returns normalized intake', async () => {
    const calls: Array<{ repo: string; issueNumber: number }> = [];

    const reader: GitHubIssueReader = {
      async getIssue(repo, issueNumber) {
        calls.push({ repo, issueNumber });
        return WORKSTREAM_ISSUE_FIXTURE;
      },
    };

    const service = new GitHubIssueIntakeService(reader);
    const intake = await service.fetchIssueIntake('Deveadra/versa', 78);

    expect(calls).toEqual([{ repo: 'Deveadra/versa', issueNumber: 78 }]);
    expect(intake.metadata.title).toContain('GitHub issue intake');
    expect(intake.requirements.dependencies).toEqual([77]);
  });
});

describe('task-card generation (WS14)', () => {
  const intake = normalizeGitHubIssue(WS14_TASK_CARD_ISSUE_FIXTURE);

  const generatorInput = {
    intake,
    workstreamId: 'WS14',
    taskCardName: 'task-card-generator',
    baseBranch: 'main',
    branch: 'orchestrator/ws14-task-card-generator',
    prTitle: 'orchestrator(ws14): add task-card generation and refresh workflow',
    dependsOn: ['#81'],
    inScope: ['Add deterministic task-card generation from normalized issue intake data'],
    outOfScope: ['Roo handoff generation'],
    filesToInspectFirst: ['docs/templates/agent-task-card.md', 'packages/integrations/src/'],
    requiredApproach: ['Inspect current repo state before editing.', 'Keep implementation additive and bounded.'],
    requiredValidation: ['pnpm install', 'pnpm lint', 'pnpm typecheck', 'pnpm test'],
    noTouchConstraints: ['Do not delete or rewrite the legacy Python runtime'],
    notesForAgent: ['This workstream is bounded to task-card generation and refresh only.'],
  };

  const model = createTaskCardRenderModel(generatorInput);

  it('builds canonical task-card filename and path', () => {
    expect(buildTaskCardFileName('WS14', 79, 'task-card-generator')).toBe(
      'ws14-issue-79-task-card-generator.md',
    );
    expect(buildTaskCardPath('WS14', 79, 'task-card-generator')).toBe(
      'docs/task-cards/active/ws14-issue-79-task-card-generator.md',
    );
  });

  it('throws when task-card slug cannot be derived', () => {
    expect(() => buildTaskCardFileName('WS14', 79, '!!!')).toThrow(
      'Could not derive a non-empty slug',
    );
  });

  it('renders required task-card sections from intake + config', () => {
    const markdown = renderTaskCardMarkdown(model);

    expect(markdown).toContain('# Agent Task Card');
    expect(markdown).toContain('- Issue: #79');
    expect(markdown).toContain('- Base Branch: main');
    expect(markdown).toContain('- Branch: orchestrator/ws14-task-card-generator');
    expect(markdown).toContain('- PR Title: orchestrator(ws14): add task-card generation and refresh workflow');
    expect(markdown).toContain('- Task Card Name: task-card-generator');
    expect(markdown).toContain('- Task Card Slug: task-card-generator');
    expect(markdown).toContain('## Objective');
    expect(markdown).toContain('## In Scope');
    expect(markdown).toContain('## Required Validation');
    expect(markdown).toContain('## No-Touch Constraints');
    expect(markdown).toContain('## Acceptance Criteria');
    expect(markdown).toContain('## Notes for Agent');
  });

  it('refresh preserves manual notes by default', () => {
    const existing = renderTaskCardMarkdown(model).replace(
      /## Notes for Agent\n\n[\s\S]*$/,
      '## Notes for Agent\n\nHuman note: keep this context.\n',
    );

    const refreshed = refreshTaskCardMarkdown(existing, {
      ...model,
      objective: 'Updated objective text',
      notesForAgent: ['Generated replacement notes'],
    });

    expect(refreshed).toContain('Updated objective text');
    expect(refreshed).toContain('Human note: keep this context.');
    expect(refreshed).not.toContain('Generated replacement notes');
  });

  it('refresh preserves manual notes for CRLF-authored cards', () => {
    const existing = renderTaskCardMarkdown(model)
      .replace(/\n/g, '\r\n')
      .replace(
        /## Notes for Agent\r\n\r\n[\s\S]*$/,
        '## Notes for Agent\r\n\r\nHuman note: keep this CRLF context.\r\n',
      );

    const refreshed = refreshTaskCardMarkdown(existing, {
      ...model,
      objective: 'Updated objective text',
      notesForAgent: ['Generated replacement notes'],
    });

    expect(refreshed).toContain('Updated objective text');
    expect(refreshed).toContain('Human note: keep this CRLF context.');
    expect(refreshed).not.toContain('Generated replacement notes');
  });

  it('renders required approach as non-empty placeholder when empty', () => {
    const markdown = renderTaskCardMarkdown({
      ...model,
      requiredApproach: [],
    });

    expect(markdown).toContain('## Required Approach');
    expect(markdown).toContain('## Required Approach\n\n- None');
  });

  it('keeps human-readable task card name and explicit slug in model', () => {
    const readableModel = createTaskCardRenderModel({
      ...generatorInput,
      taskCardName: 'Task Card Generator',
    });

    expect(readableModel.taskCardName).toBe('Task Card Generator');
    expect(readableModel.taskCardSlug).toBe('task-card-generator');
    expect(readableModel.taskCardFileName).toBe('ws14-issue-79-task-card-generator.md');
  });

  it('refresh can overwrite notes when explicitly requested', () => {
    const existing = renderTaskCardMarkdown(model).replace(
      /## Notes for Agent\n\n[\s\S]*$/,
      '## Notes for Agent\n\nHuman note: keep this context.\n',
    );

    const refreshed = refreshTaskCardMarkdown(
      existing,
      {
        ...model,
        notesForAgent: ['Generated replacement notes'],
      },
      { overwriteNotesForAgent: true },
    );

    expect(refreshed).toContain('Generated replacement notes');
    expect(refreshed).not.toContain('Human note: keep this context.');
  });
});

describe('roo handoff generation (WS15)', () => {
  const intake = normalizeGitHubIssue(WS15_ROO_HANDOFF_ISSUE_FIXTURE);

  const handoffModel = createRooHandoffRenderModel({
    intake,
    executionModeName: 'Versa Executor',
    repositoryName: 'versa',
    taskCardPath: 'docs/task-cards/active/ws15-issue-80-roo-handoff-generator.md',
    baseBranch: 'main',
    branch: 'orchestrator/ws15-roo-handoff-generator',
    objective:
      'Implement the Roo executor handoff generator so Ultron can convert a GitHub issue plus active task card into a precise Roo-ready execution prompt.',
    inScope: [
      'Add a Roo handoff contract',
      'Render a Roo-ready handoff from issue intake data and task-card data',
      'Add tests for generated handoff content',
    ],
    outOfScope: ['Live Roo dispatch', 'Result ingestion', 'Sandbox worktree creation'],
    filesToInspectFirst: [
      'docs/templates/agent-task-card.md',
      'docs/task-cards/active/',
      'packages/shared/',
      'apps/core/src/',
      'apps/ai/src/',
    ],
    requiredValidation: ['pnpm install', 'pnpm lint', 'pnpm typecheck', 'pnpm test'],
    noTouchConstraints: [
      'Do not delete or rewrite the legacy Python runtime',
      'Do not implement actual Roo dispatch in this workstream',
    ],
    expectedDeliverables: [
      'Roo handoff contract',
      'handoff renderer',
      'tests for required prompt sections',
    ],
    blockerReportingRules: [
      'Report blockers explicitly when validation fails',
      'Do not expand scope silently to fix unrelated failures',
    ],
    expectedFinalResponseFormat: [
      'files changed',
      'commands run',
      'validation results',
      'blockers, if any',
      'PR-ready summary referencing issue #80',
    ],
  });

  it('renders required Roo handoff sections from issue intake + task-card data', () => {
    const markdown = renderRooHandoffMarkdown(handoffModel);

    expect(markdown).toContain('Issue: `https://github.com/Deveadra/versa/issues/80`');
    expect(markdown).toContain('Task card: docs/task-cards/active/ws15-issue-80-roo-handoff-generator.md');
    expect(markdown).toContain('Authority order:');
    expect(markdown).toContain('1. explicit user instruction');
    expect(markdown).toContain('Issue context:');
    expect(markdown).toContain('- Base Branch: main');
    expect(markdown).toContain('- Branch: orchestrator/ws15-roo-handoff-generator');
    expect(markdown).toContain('Files/Areas to Inspect First:');
    expect(markdown).toContain('Required Validation:');
    expect(markdown).toContain('- pnpm test');
    expect(markdown).toContain('No-Touch Constraints:');
    expect(markdown).toContain('Expected Deliverables:');
    expect(markdown).toContain('Blocker Reporting Rules:');
    expect(markdown).toContain('Expected Final Response Format:');
  });

  it('matches the documented Roo handoff template example to avoid drift', () => {
    const markdown = renderRooHandoffMarkdown(handoffModel);
    const docsTemplate = readFileSync(resolve(__dirname, '../../../docs/templates/roo-handoff.md'), 'utf-8');

    expect(markdown.trim()).toBe(docsTemplate.trim());
  });
});

describe('sandbox execution preparation (WS16)', () => {
  it('builds a ready execution prep plan with required bounded context', () => {
    const result = prepareSandboxExecution({
      issueUrl: 'https://github.com/Deveadra/versa/issues/84',
      issueNumber: 84,
      taskCardPath: 'docs/task-cards/active/ws16-issue-84-sandbox-execution-prep.md',
      repoPath: '/home/devaedra/projects/versa',
      baseBranch: 'main',
      branch: 'orchestrator/ws16-sandbox-execution-prep',
      sandboxStrategy: 'git_worktree',
      validationCommands: ['pnpm install', 'pnpm lint', 'pnpm typecheck', 'pnpm test'],
      commandAllowlist: ['git status --short --branch', 'pnpm lint', 'pnpm typecheck', 'pnpm test'],
      noTouchConstraints: [
        'Do not delete or rewrite the legacy Python runtime',
        'Do not implement full Roo dispatch here',
      ],
      environmentTwinRequired: true,
      environmentTwinSlug: 'local-dev-linux',
      contextEmbedTargets: ['roo_handoff', 'run_record'],
    });

    expect(result.status).toBe('ready');
    expect(result.issues).toEqual([]);
    expect(result.plan.baseBranch).toBe('main');
    expect(result.plan.branch).toBe('orchestrator/ws16-sandbox-execution-prep');
    expect(result.plan.repoPath).toBe('/home/devaedra/projects/versa');
    expect(result.plan.sandboxStrategy).toBe('git_worktree');
    expect(result.plan.validationCommands).toContain('pnpm test');
    expect(result.plan.commandAllowlist).toContain('pnpm lint');
    expect(result.plan.noTouchConstraints).toContain('Do not implement full Roo dispatch here');
    expect(result.plan.environmentTwin).toEqual({
      required: true,
      compatible: true,
      slug: 'local-dev-linux',
    });
  });

  it('returns blocked readiness with missing required fields', () => {
    const result = prepareSandboxExecution({
      issueUrl: '',
      issueNumber: 0,
      taskCardPath: '',
      repoPath: '',
      baseBranch: '',
      branch: '',
      validationCommands: [],
      commandAllowlist: [],
      noTouchConstraints: [],
      environmentTwinRequired: true,
      environmentTwinSlug: null,
    });

    expect(result.status).toBe('blocked');
    expect(result.issues).toContain('issueUrl is required');
    expect(result.issues).toContain('issueNumber must be a positive number');
    expect(result.issues).toContain('taskCardPath is required');
    expect(result.issues).toContain('repoPath is required');
    expect(result.issues).toContain('baseBranch is required');
    expect(result.issues).toContain('branch is required');
    expect(result.issues).toContain('validationCommands must include at least one command');
    expect(result.issues).toContain('commandAllowlist must include at least one safe command');
    expect(result.issues).toContain('noTouchConstraints must include at least one boundary');
    expect(result.issues).toContain('environmentTwinSlug is required when environmentTwinRequired is true');
    expect(result.plan.environmentTwin).toEqual({
      required: true,
      compatible: false,
      slug: null,
    });
  });
});
