from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from base.quality.models import Diagnostic


@dataclass(slots=True)
class ResidualRepairRequest:
    files: tuple[Path, ...]
    diagnostics: tuple[Diagnostic, ...]
    report_dir: str | None = None
    max_files: int = 6
    max_diagnostics: int = 30


def build_residual_repair_instruction(request: ResidualRepairRequest) -> str:
    scoped_files = sorted(path.as_posix() for path in request.files)[: request.max_files]
    scoped_diagnostics = request.diagnostics[: request.max_diagnostics]

    lines = [
        "Repair the remaining quality issues in the scoped files.",
        "",
        "Constraints:",
        "- Preserve behavior.",
        "- Edit only the listed files.",
        "- Do not change architecture unless required to satisfy the diagnostics.",
        "- Prefer the smallest clean refactor that resolves the issue.",
        "- Keep imports, typing, and formatting consistent with the repo.",
        "- Do not introduce unrelated edits.",
        "",
        "Scoped files:",
    ]
    lines.extend(f"- {path}" for path in scoped_files)

    lines.extend(["", "Remaining diagnostics:"])
    for diagnostic in scoped_diagnostics:
        path = diagnostic.path.as_posix() if diagnostic.path is not None else "(unknown)"
        location = f"{path}:{diagnostic.line}" if diagnostic.line else path
        lines.append(
            f"- {diagnostic.tool.value}:{diagnostic.code} @ {location} :: {diagnostic.message}"
        )

    if request.report_dir:
        lines.extend(["", f"Repair report: {request.report_dir}"])

    return "\n".join(lines)
