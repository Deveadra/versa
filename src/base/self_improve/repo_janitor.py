from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from base.quality.models import Diagnostic, Severity, ToolName
from base.quality.policy import RepairPolicy
from base.quality.repair_service import QualityRepairService


@dataclass
class JanitorFinding:
    kind: str  # "lint" | "format" | "test" | "security" | "dead_code"
    path: str
    detail: str
    autofixable: bool
    severity: int  # 1-10


class RepoJanitor:
    def __init__(self, repo_root: str, test_runner, patcher) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.tests = test_runner
        self.patcher = patcher
        self.quality = QualityRepairService(self.repo_root)

    def scan(self) -> list[JanitorFinding]:
        policy, scope_name = self._scan_policy()
        files = self.quality.resolver.resolve_files(policy)

        snapshot = self.quality.runner.snapshot(
            files,
            run_typecheck=True,
            run_tests=False,
        )
        return self._build_findings(snapshot.diagnostics, scope_name)

    def propose_autofix(self, findings: list[JanitorFinding]) -> dict[str, Any]:
        autofixable_count = sum(1 for finding in findings if finding.autofixable)
        risk = "low" if autofixable_count == len(findings) else "medium"

        return {
            "actions": ["quality_repair_gate"],
            "scope": "changed_files",
            "risk": risk,
            "finding_count": len(findings),
            "autofixable_count": autofixable_count,
        }

    def repair_changed_files(self) -> dict[str, Any]:
        report = self.quality.repair(policy=RepairPolicy.for_changed_files())
        return report.to_dict()

    def repair_branch_delta(self, *, base_ref: str, head_ref: str = "HEAD") -> dict[str, Any]:
        report = self.quality.repair(
            policy=RepairPolicy.for_branch_delta(base_ref=base_ref, head_ref=head_ref)
        )
        return report.to_dict()

    def repair_full_repo(self) -> dict[str, Any]:
        report = self.quality.repair(policy=RepairPolicy.for_full_repo())
        return report.to_dict()

    def _scan_policy(self) -> tuple[RepairPolicy, str]:
        changed_policy = RepairPolicy.for_changed_files()
        changed_files = self.quality.resolver.resolve_files(changed_policy)
        if changed_files:
            return changed_policy, "changed_files"

        full_policy = RepairPolicy.for_full_repo()
        return full_policy, "full_repo"

    def _build_findings(
        self,
        diagnostics: tuple[Diagnostic, ...],
        scope_name: str,
    ) -> list[JanitorFinding]:
        if not diagnostics:
            return []

        grouped: dict[tuple[str, str], dict[str, Any]] = {}
        for diagnostic in diagnostics:
            key = (diagnostic.tool.value, diagnostic.code)
            entry = grouped.setdefault(
                key,
                {
                    "count": 0,
                    "path": diagnostic.path.as_posix() if diagnostic.path is not None else ".",
                    "fixable": False,
                    "severity": diagnostic.severity,
                },
            )
            entry["count"] += 1
            entry["fixable"] = (
                entry["fixable"]
                or diagnostic.fixable
                or diagnostic.code
                in {
                    "E722",
                    "B904",
                }
            )

        findings: list[JanitorFinding] = []
        for (tool_name, code), entry in sorted(grouped.items()):
            findings.append(
                JanitorFinding(
                    kind=self._kind_for_tool(tool_name),
                    path=entry["path"],
                    detail=f"{tool_name}:{code} x{entry['count']} ({scope_name})",
                    autofixable=bool(entry["fixable"]),
                    severity=self._severity_for(tool_name, entry["severity"]),
                )
            )
        return findings

    @staticmethod
    def _kind_for_tool(tool_name: str) -> str:
        if tool_name in {ToolName.RUFF.value, ToolName.PYRIGHT.value}:
            return "lint"
        if tool_name == ToolName.PYTEST.value:
            return "test"
        return "lint"

    @staticmethod
    def _severity_for(tool_name: str, severity: Severity) -> int:
        if tool_name == ToolName.PYRIGHT.value:
            return 7
        if severity is Severity.WARNING:
            return 4
        return 6
