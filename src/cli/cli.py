# src/cli/cli.py
import click

from base.agents.agent import run_goal


@click.group()
def cli() -> None:
    """Ultron CLI."""
    pass


@cli.command("goal")
@click.argument("goal")
def goal_cmd(goal: str) -> None:
    """Run agent on a high-level GOAL."""
    run_goal(goal)


if __name__ == "__main__":
    cli()
