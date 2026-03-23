from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from base.quality.diff_scope import GitDiffScopeResolver
from base.quality.policy import RepairPolicy
from base.quality.runner import QualityRunner

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify repo quality without modifying files.")
    parser.add_argument(
        "--scope",
        choices=("changed_files", "branch_delta", "full_repo"),
        default="changed_files",
        help="What to verify.",
    )
    parser.add_argument("--base", default="main", help="Base ref for branch_delta scope.")
    parser.add_argument("--head", default="HEAD", help="Head ref for branch_delta scope.")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.scope == "changed_files":
        policy = RepairPolicy.for_changed_files()
    elif args.scope == "branch_delta":
        policy = RepairPolicy.for_branch_delta(base_ref=args.base, head_ref=args.head)
    else:
        policy = RepairPolicy.for_full_repo()

    resolver = GitDiffScopeResolver(REPO_ROOT)
    runner = QualityRunner(REPO_ROOT, policy)
    files = resolver.resolve_files(policy)

    snapshot = runner.snapshot(
        files,
        run_typecheck=True,
        run_tests=False,
    )

    output_root = REPO_ROOT / "artifacts" / "quality"
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    output_path = output_root / f"verify-{timestamp}.json"
    output_path.write_text(
        json.dumps(
            {
                "scope": args.scope,
                "base_ref": args.base,
                "head_ref": args.head,
                "files_in_scope": [path.as_posix() for path in files],
                "snapshot": snapshot.to_dict(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "scope": args.scope,
                "files_in_scope": [path.as_posix() for path in files],
                "total_diagnostics": snapshot.total_diagnostics,
                "by_tool": snapshot.by_tool,
                "by_rule": snapshot.by_rule,
                "output_path": output_path.as_posix(),
            },
            indent=2,
        )
    )
    return 0 if snapshot.total_diagnostics == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
