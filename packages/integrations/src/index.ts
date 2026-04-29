export type IntegrationProvider = 'google' | 'notion' | 'github';
export const availableIntegrations: IntegrationProvider[] = ['google', 'notion', 'github'];

export interface GitHubIssueRecord {
  number: number;
  title: string;
  body?: string | null;
  state: string;
  url: string;
  labels?: Array<{ name: string } | string>;
  assignees?: Array<{ login: string } | string>;
}

export interface GitHubIssueReader {
  getIssue(repo: string, issueNumber: number): Promise<GitHubIssueRecord>;
}

export interface IssueIntakeMetadata {
  number: number;
  title: string;
  body: string;
  state: 'open' | 'closed' | 'other';
  url: string;
  labels: string[];
  assignees: string[];
}

export interface IssueIntakeRequirements {
  parentEpic: number | null;
  objective: string | null;
  why: string | null;
  deliverables: string[];
  expectedCodeChanges: string[];
  acceptanceCriteria: string[];
  constraints: string[];
  suggestedBranch: string | null;
  suggestedPrTitle: string | null;
  dependencies: number[];
  blockers: number[];
}

export interface IssueIntake {
  metadata: IssueIntakeMetadata;
  requirements: IssueIntakeRequirements;
}

const SECTION_OBJECTIVE = ['goal', 'objective'];
const SECTION_WHY = ['why'];
const SECTION_DELIVERABLES = ['deliverables'];
const SECTION_EXPECTED_CHANGES = ['expected code changes'];
const SECTION_ACCEPTANCE = ['acceptance criteria'];
const SECTION_CONSTRAINTS = ['constraints'];

function normalizeHeading(value: string): string {
  return value.trim().toLowerCase();
}

function normalizeState(value: string): 'open' | 'closed' | 'other' {
  const normalized = value.trim().toLowerCase();
  if (normalized === 'open') {
    return 'open';
  }

  if (normalized === 'closed') {
    return 'closed';
  }

  return 'other';
}

function normalizeStringList(values: Array<{ name: string } | { login: string } | string> | undefined): string[] {
  if (!values) {
    return [];
  }

  return values
    .map((entry) => {
      if (typeof entry === 'string') {
        return entry.trim();
      }

      if ('name' in entry) {
        return entry.name.trim();
      }

      if ('login' in entry) {
        return entry.login.trim();
      }

      return '';
    })
    .filter((value) => value.length > 0);
}

function parseIssueReferences(value: string | null): number[] {
  if (!value) {
    return [];
  }

  return Array.from(value.matchAll(/#(\d+)/g), (match) => Number(match[1]));
}

interface LabeledValuePattern {
  inline: RegExp;
  block: RegExp;
}

const SUGGESTED_BRANCH_PATTERNS: LabeledValuePattern = {
  inline: /Suggested\s+branch\s*:\s*(.+)/i,
  block: /^Suggested\s+branch$/i,
};

const SUGGESTED_PR_TITLE_PATTERNS: LabeledValuePattern = {
  inline: /Suggested\s+PR\s+title\s*:\s*(.+)/i,
  block: /^Suggested\s+PR\s+title$/i,
};

const LABLED_VALUE_BOUNDARY_LINES: RegExp[] = [
  /^##+\s+/,
  /^Suggested\s+branch$/i,
  /^Suggested\s+branch\s*:/i,
  /^Suggested\s+PR\s+title$/i,
  /^Suggested\s+PR\s+title\s*:/i,
  /^Depends\s+On\s*:/i,
  /^Blocks?\s*:/i,
  /^Parent\s+epic\s*:/i,
];

function isLabeledValueBoundaryLine(line: string): boolean {
  return LABLED_VALUE_BOUNDARY_LINES.some((pattern) => pattern.test(line));
}

function parseLabeledValue(body: string, patterns: LabeledValuePattern): string | null {
  const inline = body.match(patterns.inline);
  if (inline && inline[1]) {
    return inline[1].trim();
  }

  const lines = body.split(/\r?\n/);
  for (let index = 0; index < lines.length; index += 1) {
    if (patterns.block.test(lines[index].trim())) {
      for (let next = index + 1; next < lines.length; next += 1) {
        const value = lines[next].trim();
        if (isLabeledValueBoundaryLine(value)) {
          return null;
        }

        if (value.length > 0) {
          return value;
        }
      }
    }
  }

  return null;
}

function parseBulletList(value: string): string[] {
  const lines = value
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0);

  return lines
      .map((line) => line.replace(/^[-*]\s+/, '').trim())
      .filter((line) => line.length > 0);
}

function parseSectionMap(body: string): Map<string, string> {
  const sections = new Map<string, string>();
  const lines = body.split(/\r?\n/);

  let currentHeading: string | null = null;
  let currentLines: string[] = [];

  const flush = (): void => {
    if (!currentHeading) {
      return;
    }

    const value = currentLines.join('\n').trim();
    sections.set(currentHeading, value);
  };

  for (const line of lines) {
    const headingMatch = line.match(/^##+\s+(.+)$/);

    if (headingMatch) {
      flush();
      currentHeading = normalizeHeading(headingMatch[1]);
      currentLines = [];
      continue;
    }

    if (currentHeading) {
      currentLines.push(line);
    }
  }

  flush();

  return sections;
}

function firstSectionValue(sections: Map<string, string>, candidates: string[]): string | null {
  for (const name of candidates) {
    const value = sections.get(name);
    if (value && value.length > 0) {
      return value;
    }
  }

  return null;
}

export function extractIssueRequirements(body: string): IssueIntakeRequirements {
  const sections = parseSectionMap(body);
  const parentEpicMatch = body.match(/Parent\s+epic:\s*#(\d+)/i);
  const dependsOnMatch = body.match(/Depends\s+On:\s*(.+)/i);
  const blockersMatch = body.match(/Blocks?:\s*(.+)/i);

  const objectiveText = firstSectionValue(sections, SECTION_OBJECTIVE);
  const whyText = firstSectionValue(sections, SECTION_WHY);

  return {
    parentEpic: parentEpicMatch ? Number(parentEpicMatch[1]) : null,
    objective: objectiveText ? objectiveText.replace(/^\s*[-*]\s*/m, '').trim() : null,
    why: whyText ? whyText.replace(/^\s*[-*]\s*/m, '').trim() : null,
    deliverables: parseBulletList(firstSectionValue(sections, SECTION_DELIVERABLES) ?? ''),
    expectedCodeChanges: parseBulletList(firstSectionValue(sections, SECTION_EXPECTED_CHANGES) ?? ''),
    acceptanceCriteria: parseBulletList(firstSectionValue(sections, SECTION_ACCEPTANCE) ?? ''),
    constraints: parseBulletList(firstSectionValue(sections, SECTION_CONSTRAINTS) ?? ''),
    suggestedBranch: parseLabeledValue(body, SUGGESTED_BRANCH_PATTERNS),
    suggestedPrTitle: parseLabeledValue(body, SUGGESTED_PR_TITLE_PATTERNS),
    dependencies: parseIssueReferences(dependsOnMatch ? dependsOnMatch[1] : null),
    blockers: parseIssueReferences(blockersMatch ? blockersMatch[1] : null),
  };
}

export function normalizeGitHubIssue(record: GitHubIssueRecord): IssueIntake {
  const body = record.body ?? '';

  return {
    metadata: {
      number: record.number,
      title: record.title,
      body,
      state: normalizeState(record.state),
      url: record.url,
      labels: normalizeStringList(record.labels),
      assignees: normalizeStringList(record.assignees),
    },
    requirements: extractIssueRequirements(body),
  };
}

export class GitHubIssueIntakeService {
  constructor(private readonly reader: GitHubIssueReader) {}

  async fetchIssueIntake(repo: string, issueNumber: number): Promise<IssueIntake> {
    const issue = await this.reader.getIssue(repo, issueNumber);
    return normalizeGitHubIssue(issue);
  }
}

export interface TaskCardGeneratorInput {
  intake: IssueIntake;
  workstreamId: string;
  taskCardName: string;
  baseBranch: string;
  branch: string;
  prTitle: string;
  status?: string;
  priority?: string;
  agentType?: string;
  dependsOn?: string[];
  inScope: string[];
  outOfScope: string[];
  filesToInspectFirst: string[];
  requiredApproach: string[];
  requiredValidation: string[];
  noTouchConstraints: string[];
  notesForAgent: string[];
}

export interface TaskCardRenderModel {
  issueUrl: string;
  issueNumber: number;
  issueTitle: string;
  parentEpic: number | null;
  workstreamId: string;
  taskCardId: string;
  taskCardName: string;
  taskCardSlug: string;
  taskCardFileName: string;
  taskCardPath: string;
  status: string;
  priority: string;
  agentType: string;
  baseBranch: string;
  branch: string;
  prTitle: string;
  dependsOn: string[];
  objective: string;
  inScope: string[];
  outOfScope: string[];
  filesToInspectFirst: string[];
  requiredApproach: string[];
  requiredValidation: string[];
  deliverables: string[];
  noTouchConstraints: string[];
  acceptanceCriteria: string[];
  notesForAgent: string[];
}

export interface TaskCardRefreshOptions {
  overwriteNotesForAgent?: boolean;
}

export interface RooHandoffGeneratorInput {
  intake: IssueIntake;
  executionModeName: string;
  repositoryName: string;
  taskCardPath: string;
  baseBranch: string;
  branch: string;
  objective: string;
  inScope: string[];
  outOfScope: string[];
  filesToInspectFirst: string[];
  requiredValidation: string[];
  noTouchConstraints: string[];
  expectedDeliverables: string[];
  blockerReportingRules: string[];
  expectedFinalResponseFormat: string[];
}

export interface RooHandoffRenderModel {
  issueUrl: string;
  issueNumber: number;
  issueTitle: string;
  executionModeName: string;
  repositoryName: string;
  taskCardPath: string;
  baseBranch: string;
  branch: string;
  objective: string;
  inScope: string[];
  outOfScope: string[];
  filesToInspectFirst: string[];
  requiredValidation: string[];
  noTouchConstraints: string[];
  expectedDeliverables: string[];
  blockerReportingRules: string[];
  expectedFinalResponseFormat: string[];
}

export type SandboxStrategy = 'in_place_branch' | 'git_worktree' | 'dry_run_only';

export interface SandboxExecutionPrepInput {
  issueUrl: string;
  issueNumber: number;
  taskCardPath: string;
  repoPath: string;
  baseBranch: string;
  branch: string;
  validationCommands: string[];
  noTouchConstraints: string[];
  commandAllowlist: string[];
  sandboxStrategy?: SandboxStrategy;
  environmentTwinSlug?: string | null;
  environmentTwinRequired?: boolean;
  contextEmbedTargets?: Array<'roo_handoff' | 'run_record'>;
}

export interface SandboxExecutionPrepResult {
  status: 'ready' | 'blocked';
  issues: string[];
  plan: {
    issueUrl: string;
    issueNumber: number;
    taskCardPath: string;
    repoPath: string;
    baseBranch: string;
    branch: string;
    sandboxStrategy: SandboxStrategy;
    validationCommands: string[];
    commandAllowlist: string[];
    noTouchConstraints: string[];
    environmentTwin: {
      required: boolean;
      compatible: boolean;
      slug: string | null;
    };
    contextEmbedTargets: Array<'roo_handoff' | 'run_record'>;
  };
}

export type RooRunStatus = 'succeeded' | 'failed' | 'blocked' | 'partial' | 'needs-review';

export interface RooDispatchRunRecordInput {
  issueUrl: string;
  issueNumber: number;
  taskCardPath: string;
  baseBranch: string;
  branch: string;
  handoffPath: string;
  dispatchedBy: string;
  dispatchedAt?: string;
  executionMode?: string;
}

export interface RooDispatchRunRecord {
  runId: string;
  issueUrl: string;
  issueNumber: number;
  taskCardPath: string;
  baseBranch: string;
  branch: string;
  executionMode: string;
  dispatchedBy: string;
  dispatchedAt: string;
  artifacts: {
    handoffPath: string;
    outputPath: string;
    resultSummaryPath: string;
  };
}

export interface RooValidationResult {
  command: string;
  status: 'passed' | 'failed' | 'unknown';
}

export interface RooResultIngestion {
  runId: string;
  status: RooRunStatus;
  summary: string;
  validation: {
    passed: boolean;
    commands: RooValidationResult[];
  };
  changedFiles: string[];
  blockers: string[];
  rawOutput: string;
}

export interface RooChangedFilesSummary {
  total: number;
  files: string[];
}

export interface RooValidationSummary {
  passed: boolean;
  totals: {
    passed: number;
    failed: number;
    unknown: number;
  };
  commands: RooValidationResult[];
}

export interface RooResultSummary {
  runId: string;
  status: RooRunStatus;
  issue: {
    url: string | null;
    number: number | null;
  };
  taskCardPath: string | null;
  branch: string | null;
  changedFiles: RooChangedFilesSummary;
  validation: RooValidationSummary;
  blockers: string[];
  knownFollowUps: string[];
  reviewNotes: string[];
  riskMigrationNotes: string[];
  missingData: string[];
  prReadySummary: string;
}

export interface RooPrReviewPacket {
  status: RooRunStatus;
  prTitle: string | null;
  prBodyDraft: string | null;
  summary: RooResultSummary;
  missingData: string[];
}

export interface RooPostRunValidationOutcome {
  overall: 'passed' | 'failed' | 'unknown';
  passed: number;
  failed: number;
  unknown: number;
  commands: RooValidationResult[];
}

export interface RooPostRunRepoObservation {
  kind: 'changed_file' | 'review_note' | 'risk_note';
  value: string;
}

export interface RooPostRunLinkage {
  issueNumber: number | null;
  issueUrl: string | null;
  taskCardPath: string | null;
  branch: string | null;
}

export interface RooPostRunHistoryRecord {
  runId: string;
  status: RooRunStatus;
  linkage: RooPostRunLinkage;
  summary: string;
  validation: RooPostRunValidationOutcome;
  blockers: string[];
  followUps: string[];
  observations: RooPostRunRepoObservation[];
  recordedAt: string;
}

export interface RooPostRunWorkspaceUpdate {
  currentObjective: string;
  activeBlockers: Array<{
    description: string;
    status: 'active' | 'resolved';
    notes?: string;
  }>;
  importantFiles: Array<{
    path: string;
    reason: string;
  }>;
  nextRecommendedActions: Array<{
    action: string;
    priority: 'low' | 'medium' | 'high';
    rationale?: string;
  }>;
  recentDecisions: Array<{
    summary: string;
    rationale?: string;
  }>;
}

export interface RooPostRunMemoryWriteback {
  tier: 'working' | 'episodic';
  summary: string;
  content: {
    runId: string;
    status: RooRunStatus;
    linkage: RooPostRunLinkage;
    blockers: string[];
    followUps: string[];
    validation: RooPostRunValidationOutcome;
  };
  tags: string[];
}

export interface RooPostRunWorkspaceMemoryUpdate {
  runHistory: RooPostRunHistoryRecord;
  workspaceUpdate: RooPostRunWorkspaceUpdate;
  memoryWriteback: RooPostRunMemoryWriteback;
}

function buildRunId(issueNumber: number, dispatchedAt: string): string {
  return `run_${issueNumber}_${dispatchedAt.replace(/[^0-9]/g, '').slice(0, 14)}`;
}

export function createRooDispatchRunRecord(input: RooDispatchRunRecordInput): RooDispatchRunRecord {
  const dispatchedAt = input.dispatchedAt ?? new Date().toISOString();
  const runId = buildRunId(input.issueNumber, dispatchedAt);

  return {
    runId,
    issueUrl: input.issueUrl.trim(),
    issueNumber: input.issueNumber,
    taskCardPath: input.taskCardPath.trim(),
    baseBranch: input.baseBranch.trim(),
    branch: input.branch.trim(),
    executionMode: input.executionMode?.trim() || 'Versa Executor',
    dispatchedBy: input.dispatchedBy.trim(),
    dispatchedAt,
    artifacts: {
      handoffPath: input.handoffPath.trim(),
      outputPath: `artifacts/runs/${runId}/roo-output.md`,
      resultSummaryPath: `artifacts/runs/${runId}/result-summary.json`,
    },
  };
}

function parseBulletedSection(source: string, heading: string): string[] {
  const lines = source.split(/\r?\n/);
  const normalizedHeading = heading.trim().toLowerCase();
  let startIndex = -1;

  for (let index = 0; index < lines.length; index += 1) {
    const normalizedLine = lines[index].trim().toLowerCase().replace(/:$/, '');
    if (normalizedLine === normalizedHeading) {
      startIndex = index + 1;
      break;
    }
  }

  if (startIndex === -1) {
    return [];
  }

  const bullets: string[] = [];
  let encounteredBullet = false;

  for (let index = startIndex; index < lines.length; index += 1) {
    const line = lines[index].trim();
    const normalizedLine = line.toLowerCase().replace(/:$/, '');

    if (/^[-*]\s+/.test(line)) {
      encounteredBullet = true;
      bullets.push(line.replace(/^[-*]\s+/, '').trim());
      continue;
    }

    if (line.length === 0) {
      if (encounteredBullet) {
        continue;
      }

      continue;
    }

    if (!encounteredBullet && normalizedLine === heading.trim().toLowerCase()) {
      break;
    }

    if (!encounteredBullet && /^(files changed|commands run|validation results|blockers, if any|pr-ready summary)\b/.test(normalizedLine)) {
      break;
    }

    if (encounteredBullet) {
      break;
    }

    if (!encounteredBullet) {
      break;
    }
  }

  return bullets.filter((line) => line.length > 0 && line.toLowerCase() !== 'none');
}

function parseExplicitStatus(rawOutput: string): RooRunStatus | null {
  const match = rawOutput.match(/^status:\s*(succeeded|failed|blocked|partial|needs-review)\s*$/im);
  if (!match) {
    return null;
  }

  return match[1].toLowerCase() as RooRunStatus;
}

export function classifyRooRunStatus(input: {
  rawOutput: string;
  validation: RooValidationResult[];
  blockers: string[];
}): RooRunStatus {
  const explicitStatus = parseExplicitStatus(input.rawOutput);
  if (explicitStatus) {
    return explicitStatus;
  }

  const normalized = input.rawOutput.toLowerCase();

  if (input.blockers.length > 0 || normalized.includes('blocked')) {
    return 'blocked';
  }

  const hasFailedValidation = input.validation.some((row) => row.status === 'failed');
  if (hasFailedValidation || normalized.includes('validation failed') || normalized.includes('failed')) {
    return 'failed';
  }

  if (normalized.includes('partial')) {
    return 'partial';
  }

  if (normalized.includes('succeeded') || normalized.includes('completed')) {
    return 'succeeded';
  }

  return 'needs-review';
}

export function ingestRooExecutionResult(input: { runId: string; rawOutput: string }): RooResultIngestion {
  const raw = input.rawOutput;
  const summaryMatch = raw.match(/PR-ready summary:[\t ]*(.+)/i);
  const summary = summaryMatch?.[1]?.trim() ?? 'No explicit PR-ready summary found.';
  const changedFiles = parseBulletedSection(raw, 'files changed');
  const blockers = parseBulletedSection(raw, 'blockers, if any');
  const validationLines = parseBulletedSection(raw, 'validation results');

  const validation = validationLines.map((line) => {
    const lowered = line.toLowerCase();
    let status: RooValidationResult['status'] = 'unknown';
    if (lowered.includes('pass')) {
      status = 'passed';
    } else if (lowered.includes('fail')) {
      status = 'failed';
    }

    return { command: line, status };
  });

  const status = classifyRooRunStatus({
    rawOutput: raw,
    validation,
    blockers,
  });

  return {
    runId: input.runId,
    status,
    summary,
    validation: {
      passed: validation.length > 0 && validation.every((row) => row.status === 'passed'),
      commands: validation,
    },
    changedFiles,
    blockers,
    rawOutput: raw,
  };
}

function parseIssueNumberFromUrl(url: string | null): number | null {
  if (!url) return null;
  const match = url.match(/\/issues\/(\d+)/i);
  if (!match) return null;
  const parsed = Number(match[1]);
  return Number.isInteger(parsed) ? parsed : null;
}

function parseOptionalBulletedSection(source: string, heading: string): string[] {
  return parseBulletedSection(source, heading);
}

function toValidationSummary(validation: RooResultIngestion['validation']): RooValidationSummary {
  const totals = validation.commands.reduce(
    (acc, row) => {
      if (row.status === 'passed') acc.passed += 1;
      if (row.status === 'failed') acc.failed += 1;
      if (row.status === 'unknown') acc.unknown += 1;
      return acc;
    },
    { passed: 0, failed: 0, unknown: 0 },
  );

  return {
    passed: validation.passed,
    totals,
    commands: validation.commands,
  };
}

export function buildRooResultSummary(input: {
  ingestion: RooResultIngestion;
  issueUrl?: string | null;
  taskCardPath?: string | null;
  branch?: string | null;
}): RooResultSummary {
  const issueUrl = input.issueUrl?.trim() || null;
  const taskCardPath = input.taskCardPath?.trim() || null;
  const branch = input.branch?.trim() || null;

  const knownFollowUps = parseOptionalBulletedSection(input.ingestion.rawOutput, 'known follow-ups');
  const reviewNotes = parseOptionalBulletedSection(input.ingestion.rawOutput, 'review notes');
  const riskMigrationNotes = parseOptionalBulletedSection(input.ingestion.rawOutput, 'risk/migration notes');

  const missingData: string[] = [];
  if (!issueUrl) missingData.push('issueUrl');
  if (!taskCardPath) missingData.push('taskCardPath');
  if (!branch) missingData.push('branch');
  if (input.ingestion.validation.commands.length === 0) missingData.push('validation results');
  if (input.ingestion.changedFiles.length === 0) missingData.push('changed files');

  return {
    runId: input.ingestion.runId,
    status: input.ingestion.status,
    issue: {
      url: issueUrl,
      number: parseIssueNumberFromUrl(issueUrl),
    },
    taskCardPath,
    branch,
    changedFiles: {
      total: input.ingestion.changedFiles.length,
      files: [...input.ingestion.changedFiles],
    },
    validation: toValidationSummary(input.ingestion.validation),
    blockers: [...input.ingestion.blockers],
    knownFollowUps,
    reviewNotes,
    riskMigrationNotes,
    missingData,
    prReadySummary: input.ingestion.summary,
  };
}

function formatCommandRows(rows: RooValidationResult[]): string {
  if (rows.length === 0) {
    return '- none';
  }

  return rows.map((row) => `- ${row.command} (${row.status})`).join('\n');
}

function deriveWorkstreamTag(input: { summary: RooResultSummary; issueTitle: string | null }): string | null {
  const candidates = [input.summary.branch, input.summary.taskCardPath, input.issueTitle].filter(
    (value): value is string => Boolean(value),
  );

  for (const candidate of candidates) {
    const match = candidate.match(/\b(ws\d{1,2})\b/i);
    if (match) {
      return match[1].toLowerCase();
    }
  }

  return null;
}

export function buildRooPrReviewPacket(input: {
  summary: RooResultSummary;
  issueTitle?: string | null;
}): RooPrReviewPacket {
  const missingData = [...input.summary.missingData];
  const issueNumber = input.summary.issue.number;
  const issueTitle = input.issueTitle?.trim() || null;

  let prTitle: string | null = null;
  if (issueNumber && issueTitle) {
    const workstreamTag = deriveWorkstreamTag({
      summary: input.summary,
      issueTitle,
    });

    prTitle = workstreamTag
      ? `orchestrator(${workstreamTag}): ${issueTitle} (#${issueNumber})`
      : `orchestrator: ${issueTitle} (#${issueNumber})`;
  } else {
    if (!issueNumber) missingData.push('issue number for PR title');
    if (!issueTitle) missingData.push('issue title for PR title');
  }

  const canDraftBody = Boolean(input.summary.issue.url && input.summary.taskCardPath && input.summary.branch);
  if (!canDraftBody) {
    missingData.push('insufficient metadata for PR body draft');
  }

  const prBodyDraft = canDraftBody
    ? [
        `## Summary`,
        input.summary.prReadySummary,
        ``,
        `## Issue / Task Card`,
        `- Issue: ${input.summary.issue.url}`,
        `- Task Card: ${input.summary.taskCardPath}`,
        `- Branch: ${input.summary.branch}`,
        ``,
        `## Changed Files`,
        ...input.summary.changedFiles.files.map((file) => `- ${file}`),
        ``,
        `## Validation Results`,
        formatCommandRows(input.summary.validation.commands),
        ``,
        `## Blockers`,
        ...(input.summary.blockers.length > 0 ? input.summary.blockers.map((line) => `- ${line}`) : ['- none']),
        ``,
        `## Known Follow-ups`,
        ...(input.summary.knownFollowUps.length > 0
          ? input.summary.knownFollowUps.map((line) => `- ${line}`)
          : ['- none']),
        ``,
        `## Risk / Migration Notes`,
        ...(input.summary.riskMigrationNotes.length > 0
          ? input.summary.riskMigrationNotes.map((line) => `- ${line}`)
          : ['- none']),
      ].join('\n')
    : null;

  return {
    status: input.summary.status,
    prTitle,
    prBodyDraft,
    summary: input.summary,
    missingData: Array.from(new Set(missingData)),
  };
}

function deriveValidationOverall(summary: RooResultSummary): RooPostRunValidationOutcome['overall'] {
  if (summary.validation.totals.failed > 0) return 'failed';
  if (summary.validation.totals.passed > 0 && summary.validation.totals.unknown === 0) return 'passed';
  return 'unknown';
}

function toPostRunObservations(summary: RooResultSummary): RooPostRunRepoObservation[] {
  const changedFiles = summary.changedFiles.files.map((file) => ({
    kind: 'changed_file' as const,
    value: file,
  }));
  const reviewNotes = summary.reviewNotes.map((note) => ({
    kind: 'review_note' as const,
    value: note,
  }));
  const riskNotes = summary.riskMigrationNotes.map((note) => ({
    kind: 'risk_note' as const,
    value: note,
  }));
  return [...changedFiles, ...reviewNotes, ...riskNotes];
}

export function buildRooPostRunWorkspaceMemoryUpdate(input: {
  summary: RooResultSummary;
  recordedAt?: string;
}): RooPostRunWorkspaceMemoryUpdate {
  const recordedAt = input.recordedAt ?? new Date().toISOString();
  const summary = input.summary;

  const validation: RooPostRunValidationOutcome = {
    overall: deriveValidationOverall(summary),
    passed: summary.validation.totals.passed,
    failed: summary.validation.totals.failed,
    unknown: summary.validation.totals.unknown,
    commands: [...summary.validation.commands],
  };

  const linkage: RooPostRunLinkage = {
    issueNumber: summary.issue.number,
    issueUrl: summary.issue.url,
    taskCardPath: summary.taskCardPath,
    branch: summary.branch,
  };

  const blockers = [...summary.blockers];
  const followUps = [...summary.knownFollowUps];
  const observations = toPostRunObservations(summary);

  const runHistory: RooPostRunHistoryRecord = {
    runId: summary.runId,
    status: summary.status,
    linkage,
    summary: summary.prReadySummary,
    validation,
    blockers,
    followUps,
    observations,
    recordedAt,
  };

  const workspaceUpdate: RooPostRunWorkspaceUpdate = {
    currentObjective: `Post-run update for ${summary.runId} (${summary.status})`,
    activeBlockers: blockers.map((description) => ({
      description,
      status: 'active' as const,
      notes: 'Captured from executor post-run summary.',
    })),
    importantFiles: summary.changedFiles.files.map((path) => ({
      path,
      reason: 'Changed in associated run output.',
    })),
    nextRecommendedActions: followUps.map((action) => ({
      action,
      priority: 'medium' as const,
      rationale: 'Captured from known follow-ups.',
    })),
    recentDecisions: [
      {
        summary: `Run ${summary.runId} classified as ${summary.status}`,
        rationale: summary.prReadySummary,
      },
    ],
  };

  const memoryWriteback: RooPostRunMemoryWriteback = {
    tier: summary.status === 'succeeded' ? 'working' : 'episodic',
    summary: `Run ${summary.runId}: ${summary.status}`,
    content: {
      runId: summary.runId,
      status: summary.status,
      linkage,
      blockers,
      followUps,
      validation,
    },
    tags: [
      'orchestrator',
      'post-run',
      ...(summary.issue.number ? [`issue-${summary.issue.number}`] : []),
      ...(summary.branch ? [summary.branch] : []),
      `status-${summary.status}`,
    ],
  };

  return {
    runHistory,
    workspaceUpdate,
    memoryWriteback,
  };
}

function cleanList(values: unknown): string[] {
  if (!Array.isArray(values)) {
    return [];
  }

  return values
    .filter((value): value is string => typeof value === 'string')
    .map((value) => value.trim())
    .filter((value) => value.length > 0);
}

export function prepareSandboxExecution(input: SandboxExecutionPrepInput): SandboxExecutionPrepResult {
  const issues: string[] = [];
  const sandboxStrategy: SandboxStrategy = input.sandboxStrategy ?? 'in_place_branch';
  const issueUrl = input.issueUrl.trim();
  const taskCardPath = input.taskCardPath.trim();
  const repoPath = input.repoPath.trim();
  const baseBranch = input.baseBranch.trim();
  const branch = input.branch.trim();
  const validationCommands = cleanList(input.validationCommands);
  const commandAllowlist = cleanList(input.commandAllowlist);
  const noTouchConstraints = cleanList(input.noTouchConstraints);
  const contextEmbedTargets = input.contextEmbedTargets ?? ['roo_handoff', 'run_record'];
  const environmentTwinRequired = input.environmentTwinRequired ?? false;
  const environmentTwinSlug = input.environmentTwinSlug?.trim() || null;

  if (issueUrl.length === 0) {
    issues.push('issueUrl is required');
  }

  if (!Number.isFinite(input.issueNumber) || !Number.isInteger(input.issueNumber) || input.issueNumber <= 0) {
    issues.push('issueNumber must be a finite positive integer');
  }

  if (taskCardPath.length === 0) {
    issues.push('taskCardPath is required');
  }

  if (repoPath.length === 0) {
    issues.push('repoPath is required');
  }

  if (baseBranch.length === 0) {
    issues.push('baseBranch is required');
  }

  if (branch.length === 0) {
    issues.push('branch is required');
  }

  if (validationCommands.length === 0) {
    issues.push('validationCommands must include at least one command');
  }

  if (commandAllowlist.length === 0) {
    issues.push('commandAllowlist must include at least one safe command');
  }

  if (noTouchConstraints.length === 0) {
    issues.push('noTouchConstraints must include at least one boundary');
  }

  if (environmentTwinRequired && !environmentTwinSlug) {
    issues.push('environmentTwinSlug is required when environmentTwinRequired is true');
  }

  const compatibleTwin = !issues.includes('environmentTwinSlug is required when environmentTwinRequired is true');

  return {
    status: issues.length === 0 ? 'ready' : 'blocked',
    issues,
    plan: {
      issueUrl,
      issueNumber: input.issueNumber,
      taskCardPath,
      repoPath,
      baseBranch,
      branch,
      sandboxStrategy,
      validationCommands,
      commandAllowlist,
      noTouchConstraints,
      environmentTwin: {
        required: environmentTwinRequired,
        compatible: compatibleTwin,
        slug: environmentTwinSlug,
      },
      contextEmbedTargets,
    },
  };
}

function toKebabCase(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .replace(/-{2,}/g, '-');
}

function normalizeWorkstreamId(value: string): string {
  const match = value.trim().toUpperCase().match(/^WS(\d{1,2})$/);
  if (!match) {
    throw new Error(`Invalid workstreamId: ${value}. Expected format WSXX.`);
  }

  return `WS${match[1].padStart(2, '0')}`;
}

function toTaskCardSlug(taskCardName: string): string {
  const slug = toKebabCase(taskCardName);
  if (slug.length === 0) {
    throw new Error(`Invalid taskCardName: ${taskCardName}. Could not derive a non-empty slug.`);
  }

  return slug;
}

export function buildTaskCardFileName(workstreamId: string, issueNumber: number, taskCardName: string): string {
  const normalizedWs = normalizeWorkstreamId(workstreamId).toLowerCase();
  const slug = toTaskCardSlug(taskCardName);
  return `${normalizedWs}-issue-${issueNumber}-${slug}.md`;
}

export function buildTaskCardPath(workstreamId: string, issueNumber: number, taskCardName: string): string {
  return `docs/task-cards/active/${buildTaskCardFileName(workstreamId, issueNumber, taskCardName)}`;
}

export function createTaskCardRenderModel(input: TaskCardGeneratorInput): TaskCardRenderModel {
  const workstreamId = normalizeWorkstreamId(input.workstreamId);
  const issueNumber = input.intake.metadata.number;
  const taskCardSlug = toTaskCardSlug(input.taskCardName);
  const fileName = buildTaskCardFileName(workstreamId, issueNumber, input.taskCardName);

  return {
    issueUrl: input.intake.metadata.url,
    issueNumber,
    issueTitle: input.intake.metadata.title,
    parentEpic: input.intake.requirements.parentEpic,
    workstreamId,
    taskCardId: `${workstreamId}-ISSUE${issueNumber}`,
    taskCardName: input.taskCardName.trim(),
    taskCardSlug,
    taskCardFileName: fileName,
    taskCardPath: `docs/task-cards/active/${fileName}`,
    status: input.status ?? 'Active',
    priority: input.priority ?? 'High',
    agentType: input.agentType ?? 'Roo',
    baseBranch: input.baseBranch,
    branch: input.branch,
    prTitle: input.prTitle,
    dependsOn: input.dependsOn ?? [],
    objective: input.intake.requirements.objective ?? 'TBD',
    inScope: input.inScope,
    outOfScope: input.outOfScope,
    filesToInspectFirst: input.filesToInspectFirst,
    requiredApproach: input.requiredApproach,
    requiredValidation: input.requiredValidation,
    deliverables: input.intake.requirements.deliverables,
    noTouchConstraints: input.noTouchConstraints,
    acceptanceCriteria: input.intake.requirements.acceptanceCriteria,
    notesForAgent: input.notesForAgent,
  };
}

function renderBulletLines(values: string[]): string {
  if (values.length === 0) {
    return '- None';
  }

  return values.map((entry) => `- ${entry}`).join('\n');
}

function renderQuotedPaths(values: string[]): string {
  if (values.length === 0) {
    return '- `None`';
  }

  return values.map((entry) => `- \`${entry}\``).join('\n');
}

export function createRooHandoffRenderModel(input: RooHandoffGeneratorInput): RooHandoffRenderModel {
  return {
    issueUrl: input.intake.metadata.url,
    issueNumber: input.intake.metadata.number,
    issueTitle: input.intake.metadata.title,
    executionModeName: input.executionModeName,
    repositoryName: input.repositoryName,
    taskCardPath: input.taskCardPath,
    baseBranch: input.baseBranch,
    branch: input.branch,
    objective: input.objective,
    inScope: input.inScope,
    outOfScope: input.outOfScope,
    filesToInspectFirst: input.filesToInspectFirst,
    requiredValidation: input.requiredValidation,
    noTouchConstraints: input.noTouchConstraints,
    expectedDeliverables: input.expectedDeliverables,
    blockerReportingRules: input.blockerReportingRules,
    expectedFinalResponseFormat: input.expectedFinalResponseFormat,
  };
}

function renderRooHandoffTemplate(model: RooHandoffRenderModel): string {
  return `Issue: \`${model.issueUrl}\`
Task card: ${model.taskCardPath}

You are operating in ${model.executionModeName} mode for the \`${model.repositoryName}\` repository.

Required workflow:

1. Read the GitHub issue first.
2. Read the task card second.
3. Extract from the task card:
   - Base Branch
   - Branch
4. Inspect the relevant repo files before making any edits.
5. Switch to the Base Branch first.
6. If needed, update the Base Branch from its remote tracking branch.
7. If the target Branch does not exist locally, create it from the Base Branch and switch to it.
8. If the target Branch already exists locally, switch to it.
9. Summarize the minimal implementation plan before editing.
10. Execute only the assigned task card.
11. Stay strictly within scope.
12. Run every validation command listed in the task card before declaring completion.
13. Report:

- files changed
- commands run
- validation results
- blockers, if any
- a PR-ready summary referencing the issue

Authority order:

1. explicit user instruction
2. linked GitHub issue
3. active task card
4. repo-local conventions

Issue context:

- Issue URL: ${model.issueUrl}
- Issue: #${model.issueNumber}
- Issue Title: ${model.issueTitle}
- Task Card Path: ${model.taskCardPath}
- Base Branch: ${model.baseBranch}
- Branch: ${model.branch}

Objective:

${model.objective}

In Scope:

${renderBulletLines(model.inScope)}

Out of Scope:

${renderBulletLines(model.outOfScope)}

Files/Areas to Inspect First:

${renderQuotedPaths(model.filesToInspectFirst)}

Required Validation:

${renderBulletLines(model.requiredValidation)}

No-Touch Constraints:

${renderBulletLines(model.noTouchConstraints)}

Expected Deliverables:

${renderBulletLines(model.expectedDeliverables)}

Blocker Reporting Rules:

${renderBulletLines(model.blockerReportingRules)}

Expected Final Response Format:

${renderBulletLines(model.expectedFinalResponseFormat)}
`;
}

export function renderRooHandoffMarkdown(model: RooHandoffRenderModel): string {
  return renderRooHandoffTemplate(model);
}

export function renderTaskCardMarkdown(model: TaskCardRenderModel): string {
  const parentEpic = model.parentEpic === null ? 'None' : `#${model.parentEpic}`;
  const dependsOn = model.dependsOn.length > 0 ? model.dependsOn.join(', ') : 'None';

  return `# Agent Task Card

- Issue URL: ${model.issueUrl}
- Issue: #${model.issueNumber}
- Issue Title: ${model.issueTitle}
- Parent Epic: ${parentEpic}
- Workstream: ${model.workstreamId}

- Task Card ID: ${model.taskCardId}
- Task Card Name: ${model.taskCardName}
- Task Card Slug: ${model.taskCardSlug}
- Task Card File Name: ${model.taskCardFileName}
- Task Card Path: ${model.taskCardPath}

- Status: ${model.status}
- Priority: ${model.priority}
- Agent Type: ${model.agentType}

- Base Branch: ${model.baseBranch}
- Branch: ${model.branch}
- PR Title: ${model.prTitle}

- Depends On: ${dependsOn}

## Objective

${model.objective}

## In Scope

${renderBulletLines(model.inScope)}

## Out of Scope

${renderBulletLines(model.outOfScope)}

## Files/Areas to Inspect First

${renderQuotedPaths(model.filesToInspectFirst)}

## Required Approach

${renderBulletLines(model.requiredApproach)}

## Required Validation

${renderBulletLines(model.requiredValidation)}

## Deliverables

${renderBulletLines(model.deliverables)}

## No-Touch Constraints

${renderBulletLines(model.noTouchConstraints)}

## ${TASK_SECTION_HEADINGS.acceptanceCriteria}

${renderBulletLines(model.acceptanceCriteria)}

## ${TASK_SECTION_HEADINGS.notesForAgent}

${model.notesForAgent.join('\n\n')}
`;
}

const TASK_SECTION_HEADINGS = {
  acceptanceCriteria: 'Acceptance Criteria',
  notesForAgent: 'Notes for Agent',
} as const;

function buildSectionPattern(heading: string): RegExp {
  const escaped = heading.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  return new RegExp(`^##\\s+${escaped}\\r?\\n\\r?\\n([\\s\\S]*?)(?=^##\\s+|$)`, 'm');
}

function extractSection(markdown: string, heading: string): string | null {
  const pattern = buildSectionPattern(heading);
  const match = markdown.match(pattern);
  if (!match || !match[1]) {
    return null;
  }

  return match[1].trimEnd();
}

function replaceSection(markdown: string, heading: string, content: string): string {
  const pattern = buildSectionPattern(heading);
  return markdown.replace(pattern, `## ${heading}\n\n${content}\n`);
}

export function refreshTaskCardMarkdown(
  existingMarkdown: string,
  nextModel: TaskCardRenderModel,
  options: TaskCardRefreshOptions = {},
): string {
  const rendered = renderTaskCardMarkdown(nextModel);

  if (options.overwriteNotesForAgent) {
    return rendered;
  }

  const existingNotes = extractSection(existingMarkdown, TASK_SECTION_HEADINGS.notesForAgent);
  if (!existingNotes) {
    return rendered;
  }

  return replaceSection(rendered, TASK_SECTION_HEADINGS.notesForAgent, existingNotes);
}
