from base.agents.executor import execute_step
from base.agents.planner import plan_steps
from base.database.sqlite import SQLiteConn
from base.memory.store import MemoryStore
from config.config import settings


def run_goal(goal: str):
    db = SQLiteConn(settings.db_path)
    store = MemoryStore(db.conn)

    plan = plan_steps(goal)
    store.add_event(f"Goal planned: {goal}", importance=1.0, type_="goal")
    for step in plan:
        result = execute_step(step)
        store.add_event(f"[STEP] {step}\n[RESULT] {result}", importance=0.8, type_="agent_step")
