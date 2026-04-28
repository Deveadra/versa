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
  /^Suggested\s+PR\s+title$/i,
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

  const bullets = lines
    .map((line) => line.replace(/^[-*]\s+/, '').trim())
    .filter((line) => line.length > 0);

  return bullets;
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
