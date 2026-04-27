import type { DomainEvent } from '@versa/shared';
import type { AiSkillSummary } from '../../lib/api';

export type ApprovalVisibilitySnapshot = {
  governedSkillCount: number;
  requireApprovalSkillCount: number;
  approvalRelatedEventCount: number;
};

export const deriveApprovalVisibilitySnapshot = (
  skills: AiSkillSummary[],
  events: DomainEvent[],
): ApprovalVisibilitySnapshot => {
  const governedSkills = skills.filter((skill) => skill.metadata.approval.required);
  const requireApprovalEvents = events.filter((event) => {
    const payload = event.payload as Record<string, unknown>;
    return (
      event.eventType.toLowerCase().includes('approval') ||
      'approval' in payload ||
      'requiresApproval' in payload
    );
  });

  return {
    governedSkillCount: governedSkills.length,
    requireApprovalSkillCount: governedSkills.length,
    approvalRelatedEventCount: requireApprovalEvents.length,
  };
};
