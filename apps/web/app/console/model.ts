import type { DomainEvent } from '@versa/shared';
import type { AiSkillSummary } from '../../lib/api';

export type ApprovalVisibilitySnapshot = {
  governedSkillCount: number;
  approvalRelatedEventCount: number;
};

export const deriveApprovalVisibilitySnapshot = (
  skills: AiSkillSummary[],
  events: DomainEvent[],
): ApprovalVisibilitySnapshot => {
  const governedSkills = skills.filter((skill) => skill.metadata.approval.required);
  const requireApprovalEvents = events.filter((event) => {
    const payload = event.payload;
    const hasApprovalSignals =
      typeof payload === 'object' &&
      payload !== null &&
      ('approval' in payload || 'requiresApproval' in payload);

    return (
      event.eventType.toLowerCase().includes('approval') ||
      hasApprovalSignals
    );
  });

  return {
    governedSkillCount: governedSkills.length,
    approvalRelatedEventCount: requireApprovalEvents.length,
  };
};
