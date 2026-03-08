import os
from pathlib import Path
from pprint import pprint

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
    )

    result = service.run_manual(cfg=cfg, include_dream=False)
    print("=== DRY RUN RESULT ===")
    pprint(result)


if __name__ == "__main__":
    main()
