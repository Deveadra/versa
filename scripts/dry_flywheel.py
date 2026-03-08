import os
from pathlib import Path
from pprint import pprint
from typing import Any

os.environ.setdefault("GITHUB_DEFAULT_BRANCH", "feature/flywheel")

from base.database.sqlite import SQLiteConn
from base.llm.brain import Brain
from base.memory.store import MemoryStore
from base.self_improve.code_indexer import CodeIndexer
from base.self_improve.iteration_controller import IterationBudget
from base.self_improve.pr_manager import PRManager
from base.self_improve.proposal_engine import ProposalEngine
from base.self_improve.service import SelfImproveRunConfig, SelfImproveService
from config.config import settings


def _status_callback(event: dict[str, Any]) -> None:
    phase = str(event.get("phase") or "self_improve")
    state = str(event.get("state") or "update")
    pct = event.get("pct")
    iteration = event.get("iteration")
    branch = event.get("branch")
    duration_ms = event.get("duration_ms")
    outcome = event.get("outcome")
    error = event.get("error")
    message = str(event.get("message") or "")

    parts = [f"[self-improve:{phase}:{state}]"]
    if iteration is not None:
        parts.append(f"it={iteration}")
    if pct is not None:
        parts.append(f"pct={pct}")
    if branch:
        parts.append(f"branch={branch}")
    if message:
        parts.append(message)
    if outcome:
        parts.append(f"outcome={outcome}")
    if error:
        parts.append(f"error={error}")
    if duration_ms is not None:
        parts.append(f"duration_ms={int(duration_ms)}")

    print(" | ".join(parts))


def main() -> None:
    repo_root = Path(".").resolve()

    db = SQLiteConn(settings.db_path)
    store = MemoryStore(db)
    brain = Brain()

    code_indexer = CodeIndexer(
        root=str(repo_root / "src"),
        allowlist=["base", "config"],
        blocklist=["__pycache__"],
    )
    proposal_engine = ProposalEngine(repo_root=str(repo_root), brain=brain)
    pr_manager = PRManager(repo_root=str(repo_root))

    service = SelfImproveService(
        repo_root=str(repo_root),
        db=db,
        store=store,
        brain=brain,
        code_indexer=code_indexer,
        proposal_engine=proposal_engine,
        pr_manager=pr_manager,
    )

    cfg = SelfImproveRunConfig(
        budget=IterationBudget(
            max_iterations=1,
            max_seconds=300,
            stop_on_first_improvement=True,
            open_pr_on_improvement=False,
            gap_limit=3,
        ),
        gap_limit=3,
        open_pr=False,
        status_callback=_status_callback,
    )

    result = service.run_manual(cfg=cfg, include_dream=False)
    print("=== DRY RUN RESULT ===")
    pprint(result)


if __name__ == "__main__":
    main()
