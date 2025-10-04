#!/usr/bin/env python3
"""
Diagnostic scan runner for Ultron.

Features:
- Runs Black, Ruff, Pytest, and Python syntax compilation.
- Supports scanning ONLY changed files (fast) or the entire repo.
- Optional --fix to auto-apply formatting/lint fixes.
- Works on Windows/macOS/Linux (requires `git` on PATH for --changed).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def run(cmd: list[str], cwd: Path) -> tuple[int, str, str]:
    """Run a command and return (rc, stdout, stderr)."""
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            shell=False,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except FileNotFoundError as e:
        return 127, "", f"{e}"
    except Exception as e:
        return 1, "", f"{e}"


def run_cmd(cwd: Path, *cmd: str) -> tuple[int, str, str]:
    try:
        p = subprocess.run(list(cmd), check=False, cwd=str(cwd), capture_output=True, text=True)
        return p.returncode, p.stdout, p.stderr
    except Exception as e:
        return 1, "", str(e)


def try_has_xdist() -> bool:
    try:
        __import__("xdist")  # type: ignore
        return True
    except Exception:
        return False


def find_repo_root(start: Path | None = None) -> Path:
    """Walk upward until we find a directory containing .git; otherwise return start or cwd."""
    p = start or Path.cwd()
    p = p.resolve()
    for _ in range(10):
        if (p / ".git").exists():
            return p
        if p.parent == p:
            break
        p = p.parent
    return start or Path.cwd()


def git_changed_files(repo: Path, base: str | None) -> list[Path]:
    """
    Return a list of changed files.
    - If base is provided: diff base...HEAD
    - Else: working tree changes (staged, modified, untracked)
    """
    files: set[Path] = set()

    # If explicit base provided, prefer that.
    if base:
        rc, out, err = run(
            ["git", "diff", "--name-only", f"{base}...HEAD"],
            cwd=repo,
        )
        if rc == 0:
            for line in out.splitlines():
                if line.strip():
                    files.add((repo / line.strip()).resolve())
        else:
            print(f"⚠️  git diff failed against base '{base}': {err.strip()}")

    # Add current working tree changes (staged/modified/untracked)
    rc, out, err = run(["git", "status", "--porcelain"], cwd=repo)
    if rc == 0:
        for line in out.splitlines():
            if not line.strip():
                continue
            # Format examples:
            #  M path/to/file.py
            # M  path/to/file.py
            # ?? newfile.py
            path = line[3:].strip()
            if path:
                files.add((repo / path).resolve())
    else:
        print(f"⚠️  git status failed (no git?): {err.strip()}")

    return sorted([p for p in files if p.exists()])


def select_python_files(files: Iterable[Path]) -> list[Path]:
    return [f for f in files if f.suffix.lower() == ".py" and "__pycache__" not in str(f)]


def select_test_files(files: Iterable[Path]) -> list[Path]:
    tests = []
    for f in files:
        name = f.name.lower()
        if name.startswith("test_") or name.endswith("_test.py") or "tests" in f.parts:
            tests.append(f)
    return tests


def black_check(repo: Path, pyfiles: list[Path], fix: bool, scan_all: bool) -> int:
    print_section("Black")
    cmd = [sys.executable, "-m", "black"] if fix else [sys.executable, "-m", "black", "--check"]
    if scan_all or not pyfiles:
        cmd.append(".")
    else:
        cmd.extend(str(p.relative_to(repo)) for p in pyfiles)
    rc, out, err = run(cmd, cwd=repo)
    print(out.strip() or err.strip() or "(no output)")
    return rc


def ruff_check(repo: Path, pyfiles: list[Path], fix: bool, scan_all: bool) -> int:
    print_section("Ruff")
    cmd = [sys.executable, "-m", "ruff", "check"]
    if fix:
        cmd.append("--fix")
    if scan_all or not pyfiles:
        cmd.append(".")
    else:
        cmd.extend(str(p.relative_to(repo)) for p in pyfiles)
    rc, out, err = run(cmd, cwd=repo)
    print(out.strip() or err.strip() or "(no output)")
    return rc


def pytest_run(repo: Path, testfiles: list[Path], scan_all: bool, smart: bool) -> int:
    print_section("Pytest")
    # Smart skip: changed mode, no test files => skip
    if (not scan_all) and (not testfiles) and smart:
        print("No test files changed; smart skip enabled. (Use --smart-pytest to control this.)")
        return 0

    cmd = [sys.executable, "-m", "pytest", "-q"]
    if try_has_xdist():
        cmd.extend(["-n", "auto"])  # parallelize if pytest-xdist is installed
    if (not scan_all) and testfiles:
        cmd.extend(str(p.relative_to(repo)) for p in testfiles)

    rc, out, err = run(cmd, cwd=repo)
    lines = (out or err).splitlines()
    tail = "\n".join(lines[-50:]) if lines else "(no output)"
    print(tail)
    return rc


def compile_syntax(repo: Path, pyfiles: list[Path], scan_all: bool) -> int:
    print_section("Python Syntax (compileall/py_compile)")
    # If scanning all, let compileall handle it; else compile changed files individually.
    if scan_all or not pyfiles:
        rc, out, err = run([sys.executable, "-m", "compileall", "-q", "."], cwd=repo)
        print((out or err).strip() or "Syntax OK (compileall)")
        return 0 if rc == 0 else 1

    # compile changed files individually for clearer feedback
    failures = 0
    for p in pyfiles:
        rc, out, err = run(
            [
                sys.executable,
                "-c",
                f"import py_compile; py_compile.compile(r'{str(p)}', doraise=True)",
            ],
            cwd=repo,
        )
        if rc != 0:
            failures += 1
            print(f"❌ {p.relative_to(repo)}: {err.strip() or out.strip()}")
        else:
            print(f"✅ {p.relative_to(repo)}")
    return 0 if failures == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run diagnostics on the repository.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--all", action="store_true", help="Scan entire repository (default).")
    mode.add_argument(
        "--changed", action="store_true", help="Scan only changed files (requires git)."
    )
    parser.add_argument(
        "--base",
        default=None,
        help="Base ref to diff against when using --changed (e.g., origin/main).",
    )
    parser.add_argument(
        "--fix", action="store_true", help="Apply fixes (black format, ruff --fix)."
    )
    parser.add_argument(
        "--concurrent",
        action="store_true",
        help="Run independent checks (Black/Ruff) concurrently.",
    )
    parser.add_argument(
        "--smart-pytest",
        action="store_true",
        help="If --changed and no test files changed, skip pytest.",
    )
    args = parser.parse_args(argv)

    repo = find_repo_root(Path(__file__).parent)
    scan_all = args.all or (not args.changed)

    if args.changed:
        changed = git_changed_files(repo, args.base)
        py_changed = select_python_files(changed)
        tests_changed = select_test_files(py_changed)
    else:
        py_changed = []
        tests_changed = []

    print_section("Configuration")
    print(f"Repository: {repo}")
    print(f"Mode: {'ALL' if scan_all else 'CHANGED'}")
    if not scan_all:
        print(f"Base: {args.base or '<working tree>'}")
        print(f"Changed files (py): {[str(p.relative_to(repo)) for p in py_changed] or '<none>'}")
        print(f"Changed tests: {[str(p.relative_to(repo)) for p in tests_changed] or '<none>'}")
    print(f"Auto-fix: {'ON' if args.fix else 'OFF'}")

    rc_black = rc_ruff = 0
    if args.concurrent:
        print_section("Concurrent Lint/Format")
        futures = {}
        with ThreadPoolExecutor(max_workers=2) as ex:
            futures[ex.submit(black_check, repo, py_changed, fix=args.fix, scan_all=scan_all)] = (
                "black"
            )
            futures[ex.submit(ruff_check, repo, py_changed, fix=args.fix, scan_all=scan_all)] = (
                "ruff"
            )
            for fut in as_completed(futures):
                tool = futures[fut]
                try:
                    rc = fut.result()
                except Exception:
                    rc = 1
                if tool == "black":
                    rc_black = rc
                else:
                    rc_ruff = rc
    else:
        rc_black = black_check(repo, py_changed, fix=args.fix, scan_all=scan_all)
        rc_ruff = ruff_check(repo, py_changed, fix=args.fix, scan_all=scan_all)

    rc_py = pytest_run(repo, tests_changed, scan_all=scan_all, smart=args.smart_pytest)
    rc_comp = compile_syntax(repo, py_changed, scan_all=scan_all)

    print_section("Summary")
    print(f"Black:  {'OK' if rc_black == 0 else 'Issues'}")
    print(f"Ruff:   {'OK' if rc_ruff  == 0 else 'Issues'}")
    print(f"Pytest: {'OK' if rc_py    == 0 else 'Issues'}")
    print(f"Syntax: {'OK' if rc_comp  == 0 else 'Issues'}")

    # Exit non-zero if anything failed
    exit_code = 0
    for rc in (rc_black, rc_ruff, rc_py, rc_comp):
        if rc != 0:
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
