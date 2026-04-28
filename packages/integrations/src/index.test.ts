import { describe, expect, it } from 'vitest';

import {
  extractIssueRequirements,
  GitHubIssueIntakeService,
  normalizeGitHubIssue,
  type GitHubIssueReader,
} from './index';
import {
  EPIC_ISSUE_FIXTURE,
  MINIMAL_ISSUE_FIXTURE,
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
