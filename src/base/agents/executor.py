# base/agents/planner.py
from __future__ import annotations

from base.agents.tools.shell import run_shell
from base.agents.tools.memory import recall_memory, write_memory
from base.agents.tools.files import read_file, write_file

def execute_step(step: str) -> str:
    # Extremely simple router
    if "terminal" in step or "shell" in step:
        return run_shell(step)
    if "memory" in step and "recall" in step:
        return recall_memory(step)
    if "memory" in step and "write" in step:
        return write_memory(step)
    if "file" in step and "read" in step:
        return read_file(step)
    if "file" in step and "write" in step:
        return write_file(step)
    return f"[UNHANDLED] {step}"
