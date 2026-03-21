from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from base.quality.models import RepairReport


def write_repair_report(report: RepairReport, output_root: Path) -> Path:
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    report_dir = output_root / f"repair-{timestamp}"
    report_dir.mkdir(parents=True, exist_ok=True)

    json_path = report_dir / "repair_report.json"
    json_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    markdown_path = report_dir / "repair_report.md"
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")

    return report_dir


def _render_markdown(report: RepairReport) -> str:
    baseline_total = report.baseline.total_diagnostics
    after_autofix_total = report.after_autofix.total_diagnostics
    final_total = report.final_snapshot.total_diagnostics

    lines = [
        "# Quality Repair Report",
        "",
        f"- Scope: `{report.scope_name}`",
        f"- Files in scope: `{len(report.files_in_scope)}`",
        f"- Baseline diagnostics: `{baseline_total}`",
        f"- After auto-fix diagnostics: `{after_autofix_total}`",
        f"- Final diagnostics: `{final_total}`",
        f"- Success: `{report.success}`",
        "",
        "## Files in scope",
        "",
    ]

    if report.files_in_scope:
        lines.extend(f"- `{path.as_posix()}`" for path in report.files_in_scope)
    else:
        lines.append("- None")

    lines.extend(["", "## Repair rounds", ""])

    if report.rounds:
        for round_ in report.rounds:
            lines.append(f"### Round {round_.number}")
            lines.append("")
            lines.append(f"- Before: `{round_.snapshot_before.total_diagnostics}` diagnostics")
            lines.append(f"- After: `{round_.snapshot_after.total_diagnostics}` diagnostics")
            lines.append("")
            for repair in round_.repairs:
                lines.append(f"- `{repair.path.as_posix()}`")
                lines.append(f"  - changed: `{repair.changed}`")
                lines.append(
                    f"  - rules: `{', '.join(repair.rules_applied) if repair.rules_applied else 'none'}`"
                )
                if repair.notes:
                    for note in repair.notes:
                        lines.append(f"  - note: {note}")
            lines.append("")
    else:
        lines.append("- No strategy rounds were applied.")
        lines.append("")

    lines.extend(["## Remaining diagnostics", ""])

    if report.blocked:
        for diagnostic in report.blocked:
            location = ""
            if diagnostic.path is not None and diagnostic.line is not None:
                location = f" ({diagnostic.path.as_posix()}:{diagnostic.line})"
            elif diagnostic.path is not None:
                location = f" ({diagnostic.path.as_posix()})"

            lines.append(
                f"- `{diagnostic.tool.value}:{diagnostic.code}`{location} {diagnostic.message}"
            )
    else:
        lines.append("- None")

    if report.notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in report.notes)

    lines.append("")
    return "\n".join(lines)
