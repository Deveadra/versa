import { randomUUID } from 'node:crypto';
import {
  SkillDefinitionSchema,
  SkillExecutionRequestSchema,
  SkillExecutionResultSchema,
  type SkillDefinition,
  type SkillExecutionRequest,
  type SkillExecutionResult,
} from '@versa/shared';
import { ZodError } from 'zod';

export type SkillHandler = (input: Record<string, unknown>) => Record<string, unknown>;

export type SkillRegistry = {
  register(definition: SkillDefinition, handler: SkillHandler): void;
  list(): SkillDefinition[];
  get(skillIdOrName: string): SkillDefinition | null;
  execute(request: SkillExecutionRequest): SkillExecutionResult;
};

const nowIso = () => new Date().toISOString();

const errorMessage = (error: unknown, fallback: string) => {
  if (error instanceof ZodError) {
    return error.issues.map((issue: { message: string }) => issue.message).join('; ');
  }
  if (error instanceof Error) {
    return error.message;
  }
  return fallback;
};

export const createSkillRegistry = (
  options?: {
    now?: () => string;
    executionIdFactory?: () => string;
  },
): SkillRegistry => {
  const now = options?.now ?? nowIso;
  const executionIdFactory = options?.executionIdFactory ?? (() => `skx_${randomUUID().slice(0, 8)}`);
  const byId = new Map<string, SkillDefinition>();
  const byName = new Map<string, SkillDefinition>();
  const handlers = new Map<string, SkillHandler>();

  const register = (definition: SkillDefinition, handler: SkillHandler) => {
    const parsed = SkillDefinitionSchema.parse(definition);
    const existing = byId.get(parsed.id);
    if (existing && existing.name !== parsed.name) {
      byName.delete(existing.name);
    }
    byId.set(parsed.id, parsed);
    byName.set(parsed.name, parsed);
    handlers.set(parsed.id, handler);
  };

  const list = () => Array.from(byId.values());

  const resolveByName = (name: string): SkillDefinition | null => byName.get(name) ?? null;

  const get = (skillIdOrName: string): SkillDefinition | null =>
    byId.get(skillIdOrName) ?? resolveByName(skillIdOrName);

  const execute = (request: SkillExecutionRequest): SkillExecutionResult => {
    const startedAt = now();
    let parsed: SkillExecutionRequest;

    try {
      parsed = SkillExecutionRequestSchema.parse(request);
    } catch (error) {
      return SkillExecutionResultSchema.parse({
        executionId: executionIdFactory(),
        skillId: 'unknown',
        skillName: 'unknown',
        status: 'invalid_request',
        startedAt,
        completedAt: now(),
        output: {},
        validation: {
          passed: false,
          checks: [
            {
              id: 'request.valid',
              passed: false,
              message: errorMessage(error, 'invalid skill execution request'),
            },
          ],
        },
        error: {
          code: 'INVALID_REQUEST',
          message: errorMessage(error, 'invalid skill execution request'),
        },
      });
    }

    const resolved =
      parsed.skillId !== undefined && parsed.skillId.length > 0
        ? byId.get(parsed.skillId) ?? null
        : parsed.skillName
          ? resolveByName(parsed.skillName)
          : null;

    if (!resolved) {
      return SkillExecutionResultSchema.parse({
        executionId: executionIdFactory(),
        skillId: parsed.skillId ?? 'unknown',
        skillName: parsed.skillName ?? 'unknown',
        status: 'failed',
        startedAt,
        completedAt: now(),
        output: {},
        validation: {
          passed: false,
          checks: [
            {
              id: 'skill.exists',
              passed: false,
              message: 'requested skill is not registered',
            },
          ],
        },
        error: {
          code: 'SKILL_NOT_FOUND',
          message: 'requested skill is not registered',
        },
      });
    }

    const handler = handlers.get(resolved.id);
    if (!handler) {
      return SkillExecutionResultSchema.parse({
        executionId: executionIdFactory(),
        skillId: resolved.id,
        skillName: resolved.name,
        status: 'blocked',
        startedAt,
        completedAt: now(),
        output: {},
        validation: {
          passed: false,
          checks: [
            {
              id: 'skill.handler.present',
              passed: false,
              message: 'skill has no handler registered',
            },
          ],
        },
        error: {
          code: 'SKILL_HANDLER_MISSING',
          message: 'skill has no handler registered',
        },
      });
    }

    try {
      const output = handler(parsed.input);
      return SkillExecutionResultSchema.parse({
        executionId: executionIdFactory(),
        skillId: resolved.id,
        skillName: resolved.name,
        status: 'succeeded',
        startedAt,
        completedAt: now(),
        output,
        validation: {
          passed: true,
          checks: resolved.metadata.validationChecks.map((check) => ({
            id: check.id,
            passed: true,
            message: check.description,
          })),
        },
      });
    } catch (error) {
      return SkillExecutionResultSchema.parse({
        executionId: executionIdFactory(),
        skillId: resolved.id,
        skillName: resolved.name,
        status: 'failed',
        startedAt,
        completedAt: now(),
        output: {},
        validation: {
          passed: false,
          checks: [
            {
              id: 'skill.execution',
              passed: false,
              message: error instanceof Error ? error.message : 'skill execution failed',
            },
          ],
        },
        error: {
          code: 'SKILL_EXECUTION_FAILED',
          message: error instanceof Error ? error.message : 'skill execution failed',
        },
      });
    }
  };

  return {
    register,
    list,
    get,
    execute,
  };
};

const fixedFailureHandling = {
  retryable: false,
  maxRetries: 0,
} as const;

const noApproval = {
  required: false,
} as const;

export const createFoundationalSkills = (): SkillDefinition[] => [
  {
    id: 'repo-inspection',
    name: 'repo_inspection',
    bounded: true,
    deterministic: true,
    tags: ['inspection', 'repo'],
    metadata: {
      description: 'Produce a bounded repository inspection summary from explicit inputs.',
      version: '0.1.0',
      inputs: [
        {
          name: 'files',
          description: 'List of relevant files and directories already discovered.',
          required: true,
          schemaHint: 'string[]',
        },
      ],
      outputs: [
        {
          name: 'summary',
          description: 'Bounded summary of discovered repository state.',
          schemaHint: '{ summary: string; files: string[] }',
        },
      ],
      requiredTools: ['read_file', 'list_files'],
      requiredResources: ['workspace'],
      validationChecks: [
        {
          id: 'inputs.files.present',
          description: 'inputs include at least one relevant path',
          required: true,
        },
      ],
      failureHandling: fixedFailureHandling,
      approval: noApproval,
    },
  },
  {
    id: 'issue-branch-prep',
    name: 'issue_branch_prep',
    bounded: true,
    deterministic: true,
    tags: ['git', 'issue'],
    metadata: {
      description: 'Prepare issue-to-branch execution context without side effects.',
      version: '0.1.0',
      inputs: [
        {
          name: 'issue',
          description: 'Issue metadata including number/url and branch naming intent.',
          required: true,
          schemaHint: '{ number: number; url: string; suggestedBranch?: string }',
        },
      ],
      outputs: [
        {
          name: 'prep',
          description: 'Resolved prep summary for branch execution.',
          schemaHint: '{ issueNumber: number; branch: string }',
        },
      ],
      requiredTools: ['execute_command'],
      requiredResources: ['git'],
      validationChecks: [
        {
          id: 'issue.number.present',
          description: 'issue number is present and valid',
          required: true,
        },
      ],
      failureHandling: fixedFailureHandling,
      approval: noApproval,
    },
  },
  {
    id: 'pr-summary-generation',
    name: 'pr_summary_generation',
    bounded: true,
    deterministic: true,
    tags: ['pr', 'report'],
    metadata: {
      description: 'Generate a bounded PR summary from explicit changed-file and validation inputs.',
      version: '0.1.0',
      inputs: [
        {
          name: 'changes',
          description: 'List of changed files and concise notes.',
          required: true,
          schemaHint: '{ files: string[]; notes: string[] }',
        },
      ],
      outputs: [
        {
          name: 'summary',
          description: 'PR-ready markdown summary.',
          schemaHint: '{ body: string }',
        },
      ],
      requiredTools: [],
      requiredResources: ['issue-context'],
      validationChecks: [
        {
          id: 'changes.files.present',
          description: 'changed files are present',
          required: true,
        },
      ],
      failureHandling: fixedFailureHandling,
      approval: noApproval,
    },
  },
  {
    id: 'baseline-report-generation',
    name: 'baseline_report_generation',
    bounded: true,
    deterministic: true,
    tags: ['baseline', 'report'],
    metadata: {
      description: 'Generate a baseline validation report from command output snapshots.',
      version: '0.1.0',
      inputs: [
        {
          name: 'validations',
          description: 'Validation command outputs grouped by command.',
          required: true,
          schemaHint: '{ commands: Array<{ command: string; status: string }> }',
        },
      ],
      outputs: [
        {
          name: 'report',
          description: 'Baseline report structure for operational review.',
          schemaHint: '{ passed: boolean; commands: Array<{ command: string; status: string }> }',
        },
      ],
      requiredTools: [],
      requiredResources: ['validation-output'],
      validationChecks: [
        {
          id: 'validations.commands.present',
          description: 'at least one validation command result exists',
          required: true,
        },
      ],
      failureHandling: fixedFailureHandling,
      approval: noApproval,
    },
  },
];

const ensureArray = (value: unknown): string[] =>
  Array.isArray(value) ? value.map((item) => String(item)) : [];

export const createFoundationalSkillHandlers = (): Record<string, SkillHandler> => ({
  'repo-inspection': (input) => {
    const files = ensureArray(input.files);
    if (files.length === 0) {
      throw new Error('files input is required for repo inspection');
    }
    return {
      summary: `inspected ${files.length} repository path(s)`,
      files,
    };
  },
  'issue-branch-prep': (input) => {
    const issue = (input.issue ?? {}) as Record<string, unknown>;
    const number = Number(issue.number);
    if (!Number.isInteger(number) || number < 1) {
      throw new Error('issue.number must be a positive integer');
    }
    const suggestedBranch =
      typeof issue.suggestedBranch === 'string' && issue.suggestedBranch.length > 0
        ? issue.suggestedBranch
        : `issue/${number}`;
    return {
      issueNumber: number,
      branch: suggestedBranch,
      issueUrl: typeof issue.url === 'string' ? issue.url : undefined,
    };
  },
  'pr-summary-generation': (input) => {
    const changes = (input.changes ?? {}) as Record<string, unknown>;
    const files = ensureArray(changes.files);
    if (files.length === 0) {
      throw new Error('changes.files must include at least one file');
    }
    const notes = ensureArray(changes.notes);
    const lines = [
      `Files changed (${files.length}):`,
      ...files.map((file) => `- ${file}`),
      ...(notes.length > 0 ? ['', 'Notes:', ...notes.map((note) => `- ${note}`)] : []),
    ];
    return { body: lines.join('\n') };
  },
  'baseline-report-generation': (input) => {
    const validations = (input.validations ?? {}) as Record<string, unknown>;
    const commandsRaw = Array.isArray(validations.commands) ? validations.commands : [];
    const commands = commandsRaw.map((item) => {
      const row = item as Record<string, unknown>;
      return {
        command: String(row.command ?? ''),
        status: String(row.status ?? 'unknown'),
      };
    });

    if (commands.length === 0) {
      throw new Error('validations.commands must contain at least one command result');
    }

    return {
      passed: commands.every((row) => row.status === 'passed'),
      commands,
    };
  },
});

export const createFoundationalSkillRegistry = (): SkillRegistry => {
  const registry = createSkillRegistry();
  const skills = createFoundationalSkills();
  const handlers = createFoundationalSkillHandlers();

  for (const definition of skills) {
    const handler = handlers[definition.id];
    if (!handler) {
      throw new Error(`missing handler for foundational skill: ${definition.id}`);
    }
    registry.register(definition, handler);
  }

  return registry;
};

export type { SkillDefinition, SkillExecutionRequest, SkillExecutionResult } from '@versa/shared';
