import { describe, expect, it } from 'vitest';
import { createFoundationalSkillRegistry, createSkillRegistry } from './index';

describe('createSkillRegistry', () => {
  it('registers and executes a deterministic skill through one shared execution path', () => {
    const registry = createSkillRegistry({
      now: () => '2026-01-01T00:00:00.000Z',
      executionIdFactory: () => 'skx_fixed_1',
    });

    registry.register(
      {
        id: 'echo',
        name: 'echo',
        bounded: true,
        deterministic: true,
        tags: ['test'],
        metadata: {
          description: 'echo input',
          version: '0.1.0',
          inputs: [{ name: 'value', description: 'value to echo', required: true }],
          outputs: [{ name: 'value', description: 'echoed value' }],
          requiredTools: [],
          requiredResources: [],
          validationChecks: [
            { id: 'input.value.present', description: 'value was provided', required: true },
          ],
          failureHandling: { retryable: false, maxRetries: 0 },
          approval: { required: false },
        },
      },
      (input) => ({ value: input.value }),
    );

    const result = registry.execute({
      skillName: 'echo',
      input: { value: 'hello' },
      context: {},
    });

    expect(result.executionId).toBe('skx_fixed_1');
    expect(result.status).toBe('succeeded');
    expect(result.output.value).toBe('hello');
    expect(result.validation.passed).toBe(true);
  });

  it('returns failed result when skill is not found', () => {
    const registry = createSkillRegistry({
      now: () => '2026-01-01T00:00:00.000Z',
      executionIdFactory: () => 'skx_fixed_2',
    });

    const result = registry.execute({
      skillId: 'missing',
      input: {},
      context: {},
    });

    expect(result.status).toBe('failed');
    expect(result.error?.code).toBe('SKILL_NOT_FOUND');
  });
});

describe('createFoundationalSkillRegistry', () => {
  it('registers the ws06 foundational skills and can execute one by name', () => {
    const registry = createFoundationalSkillRegistry();
    const skillNames = registry.list().map((item) => item.name);

    expect(skillNames).toContain('repo_inspection');
    expect(skillNames).toContain('issue_branch_prep');
    expect(skillNames).toContain('pr_summary_generation');
    expect(skillNames).toContain('baseline_report_generation');

    const result = registry.execute({
      skillName: 'repo_inspection',
      input: {
        files: ['package.json', 'packages/shared/src/index.ts'],
      },
      context: {},
    });

    expect(result.status).toBe('succeeded');
    expect(result.output.summary).toContain('inspected 2 repository path(s)');
  });
});
