
# In cli.py
from base.agents.agent import run_goal

@cli.command()
@click.argument("goal")
def goal(goal):
    """Run agent on a high-level GOAL"""
    run_goal(goal)
