from __future__ import annotations

import errno
from pathlib import Path

from base.quality.policy import RepairPolicy
from base.quality.runner import QualityRunner


def test_run_command_oserror_returns_failed_command(monkeypatch, tmp_path: Path):
    policy = RepairPolicy.for_changed_files()
    runner = QualityRunner(tmp_path, policy)

    def fake_run(*args, **kwargs):
        raise OSError(errno.ENODEV, "No such device", "pnpm")

    monkeypatch.setattr("subprocess.run", fake_run)

    command = runner.run_prettier_check((Path("apps/web/src/example.ts"),))[0]

    assert command.returncode == 126
    assert "OSError" in command.stderr
    assert "pnpm" in command.stderr
