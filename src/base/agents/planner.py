# base/agents/planner.py
from __future__ import annotations

from base.llm.brain import ask_brain


def plan_steps(goal: str) -> list[str]:
    prompt = (
        f"You are Ultron's autonomous planning module. Break the following user goal into clear, numbered steps "
        f"that an agent can execute with tools like memory access, file reading, writing, shell, web browsing, etc.\n\n"
        f"Goal: {goal}\n\nSteps:\n"
    )
    response = ask_brain(prompt, system_prompt="Planner", response_format="text")
    steps = [line.strip() for line in response.split("\n") if line.strip() and line[0].isdigit()]
    return steps
