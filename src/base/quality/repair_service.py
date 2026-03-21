from __future__ import annotations

from pathlib import Path

from base.quality.diff_scope import GitDiffScopeResolver
from base.quality.models import Diagnostic, QualitySnapshot, RepairReport, RepairRound
from base.quality.policy import RepairPolicy
from base.quality.report import write_repair_report
from base.quality.runner import QualityRunner
from base.quality.strategies import SafeStrategyApplier


class QualityRepairService:
    def __init__(
        self,
        repo_root: Path,
        *,
        runner: QualityRunner | None = None,
        resolver: GitDiffScopeResolver | None = None,
        strategy_applier: SafeStrategyApplier | None = None,
        policy: RepairPolicy | None = None,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.policy = policy or RepairPolicy()
        self.runner = runner or QualityRunner(self.repo_root, self.policy)
        self.resolver = resolver or GitDiffScopeResolver(self.repo_root)
        self.strategy_applier = strategy_applier or SafeStrategyApplier(self.repo_root)

    def repair(
        self,
        *,
        policy: RepairPolicy | None = None,
        explicit_files: tuple[Path, ...] = (),
    ) -> RepairReport:
        active_policy = policy or self.policy
        files = self.resolver.resolve_files(active_policy, explicit_files=explicit_files)

        if not files:
            empty_snapshot = QualitySnapshot()
            report = RepairReport(
                repo_root=self.repo_root,
                scope_name=active_policy.scope_mode.value,
                files_in_scope=(),
                baseline=empty_snapshot,
                after_autofix=empty_snapshot,
                final_snapshot=empty_snapshot,
                rounds=(),
                blocked=(),
                notes=("No files found in scope.",),
            )
            report_dir = write_repair_report(report, self.repo_root / active_policy.report_root)
            report.report_dir = report_dir
            return report

        baseline = self.runner.snapshot(
            files,
            run_typecheck=active_policy.run_typecheck,
            run_tests=False,
        )

        if active_policy.auto_format:
            self.runner.run_ruff_format(files, fix=True)
        if active_policy.auto_fix_ruff:
            self.runner.run_ruff_check(files, fix=True)

        after_autofix = self.runner.snapshot(
            files,
            run_typecheck=active_policy.run_typecheck,
            run_tests=False,
        )

        rounds: list[RepairRound] = []
        notes: list[str] = []
        current_snapshot = after_autofix

        for round_number in range(1, active_policy.max_rounds + 1):
            actionable = tuple(
                diagnostic
                for diagnostic in current_snapshot.diagnostics
                if diagnostic.path is not None
                and diagnostic.path in files
                and diagnostic.code in active_policy.allowed_rule_codes
            )
            if not actionable:
                break

            backup_paths = sorted(
                {diagnostic.path for diagnostic in actionable if diagnostic.path is not None}
            )
            backups = {
                path: (self.repo_root / path).read_text(encoding="utf-8") for path in backup_paths
            }

            repairs = self.strategy_applier.apply(
                actionable,
                max_repairs=active_policy.max_repairs_per_round,
            )
            if not any(repair.changed for repair in repairs):
                notes.append(
                    "No additional safe strategy repairs could be applied to the remaining diagnostics."
                )
                break

            next_snapshot = self.runner.snapshot(
                files,
                run_typecheck=active_policy.run_typecheck,
                run_tests=False,
            )

            if next_snapshot.total_diagnostics > current_snapshot.total_diagnostics:
                for path, content in backups.items():
                    (self.repo_root / path).write_text(content, encoding="utf-8")
                reverted_snapshot = self.runner.snapshot(
                    files,
                    run_typecheck=active_policy.run_typecheck,
                    run_tests=False,
                )
                notes.append(f"Round {round_number} was reverted because diagnostics increased.")
                current_snapshot = reverted_snapshot
                break

            rounds.append(
                RepairRound(
                    number=round_number,
                    repairs=repairs,
                    snapshot_before=current_snapshot,
                    snapshot_after=next_snapshot,
                )
            )
            current_snapshot = next_snapshot

        final_snapshot = self.runner.snapshot(
            files,
            run_typecheck=active_policy.run_typecheck,
            run_tests=active_policy.run_tests,
            pytest_args=active_policy.pytest_args,
        )

        blocked = tuple(
            diagnostic
            for diagnostic in final_snapshot.diagnostics
            if self._is_in_scope(diagnostic, files)
        )

        report = RepairReport(
            repo_root=self.repo_root,
            scope_name=active_policy.scope_mode.value,
            files_in_scope=files,
            baseline=baseline,
            after_autofix=after_autofix,
            final_snapshot=final_snapshot,
            rounds=tuple(rounds),
            blocked=blocked,
            notes=tuple(notes),
        )

        report_dir = write_repair_report(report, self.repo_root / active_policy.report_root)
        report.report_dir = report_dir
        return report

    @staticmethod
    def _is_in_scope(diagnostic: Diagnostic, files: tuple[Path, ...]) -> bool:
        return diagnostic.path is not None and diagnostic.path in files
