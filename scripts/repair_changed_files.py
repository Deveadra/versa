from __future__ import annotations

import json
import sys
from pathlib import Path

from base.quality.policy import RepairPolicy
from base.quality.repair_service import QualityRepairService

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def main() -> int:
    service = QualityRepairService(REPO_ROOT)
    policy = RepairPolicy.for_changed_files()
    policy.run_typecheck = True
    policy.run_tests = False

    report = service.repair(policy=policy)

    summary = {
        "scope": "changed_files",
        "success": report.success,
        "files_in_scope": [path.as_posix() for path in report.files_in_scope],
        "baseline_total": report.baseline.total_diagnostics,
        "after_autofix_total": report.after_autofix.total_diagnostics,
        "final_total": report.final_snapshot.total_diagnostics,
        "blocked_count": len(report.blocked),
        "report_dir": report.report_dir.as_posix() if report.report_dir is not None else None,
    }
    print(json.dumps(summary, indent=2))
    return 0 if report.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
