Assumptions:

You’re running from the repo root.
You have Python 3.11 and git installed.
You have OPENAI_API_KEY set if you want proposal generation to be LLM-driven.

1. Create an isolated branch for this run
```bash
git checkout -b repo-janitor/milestone-a
```

2. Install dev dependencies (the repo already documents using .[dev])
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

3. Run baseline diagnostics (human-readable)
```bash
python scripts/diagnostic_scan.py --all
```

Expected output: sections for Black, Ruff, Pytest, Syntax, plus a Summary.

4. Run baseline scoreboard (machine-readable)
```bash
python -c "from base.self_improve.scoreboard import ScoreboardRunner; import json; r=ScoreboardRunner('.').run(); print(r.score());"
```

Expected output: a numeric score and logs indicating failing gates.

5. Run the 20-minute iteration controller
```bash
python -c "
from base.self_improve.iteration_controller import RepoJanitorController, IterationBudget
ctl=RepoJanitorController('.')
out=ctl.run(goal='Repo Janitor v1', budget=IterationBudget(max_seconds=1200, max_iters=8), open_pr=False)
print(out)
"
```

Expected outputs:

- Per-iteration logs showing improved vs reverted attempts.
- A final object like:
  baseline.score, best.score
  attempts[] with improved flags
  (optional) pr_url if you run with promotion enabled.

6. Rollback / safety exits
If the working tree got messy or you want to fully revert the attempt:

```bash
git reset --hard HEAD
git clean -fd
git checkout main
git branch -D repo-janitor/milestone-a
```


If a branch was pushed or PR opened, close it manually (human approval gate) unless you explicitly want auto-cleanup.