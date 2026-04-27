# Aerith / Ult MVP (Local, Modular, Upgradeable)

Aerith (Ult MVP) is a modular assistant with:
- SQLite-backed memory (events + facts + optional KG tables)
- Retrieval (FAISS by default; optional Qdrant client)
- Optional voice mode (STT + TTS)
- Optional Google integrations (Calendar + Gmail)
- Optional API mode (FastAPI)

This repo uses **pyproject.toml as the source of truth** for dependencies. `requirements*.txt` are wrappers only.

---

## Table of contents
- [Supported Python + OS](#supported-python--os)
- [Blank-slate setup (recommended)](#blank-slate-setup-recommended)
- [Manual install (fallback)](#manual-install-fallback)
- [Configuration (.env)](#configuration-env)
- [Run](#run)
- [Install profiles (extras)](#install-profiles-extras)
- [Voice setup (optional)](#voice-setup-optional)
- [Google Calendar + Gmail (optional)](#google-calendar--gmail-optional)
- [API mode (optional)](#api-mode-optional)
- [Testing + linting](#testing--linting)
- [Troubleshooting](#troubleshooting)

---

## Supported Python + OS

### Python
- Recommended: **Python 3.11 or 3.12**
- Voice STT behavior:
  - **Python < 3.12** installs `openai-whisper`
  - **Python >= 3.12** uses `faster-whisper` (preferred)

### OS
- Best supported: Linux / Ubuntu / WSL2
- macOS and Windows are supported, but voice/audio dependencies are more sensitive.

---

## Blank-slate setup (recommended)

This is the “works from nothing” path. It:
- deletes and recreates `.venv`
- installs dependencies via `pyproject.toml`
- runs `pytest -q` as a sanity check

> IMPORTANT: Run bootstrap from a **fresh shell** that is **NOT already inside** `.venv`.
> Bootstrap intentionally deletes `.venv`.

### Ubuntu / Debian / WSL
```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git

./scripts/bootstrap.sh
```

### Windows
```bash
.\scripts\bootstrap.ps1
```

---

## Manual Install

### macOS / Linux / WSL
```bash
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip wheel
python -m pip install --upgrade "setuptools<82"

# Dev install (recommended)
python -m pip install -e ".[dev]"

# Optional: copy env sample if present
cp .env.sample .env 2>/dev/null || true

pytest -q
python run.py
```


### Windows (PowerShell)
```shell
Remove-Item -Recurse -Force .venv -ErrorAction SilentlyContinue

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip wheel
python -m pip install --upgrade "setuptools<82"

python -m pip install -e ".[dev]"

Copy-Item .env.sample .env -ErrorAction SilentlyContinue
# If .env.sample does not exist, create .env manually.

pytest -q
python run.py
```

## Run
```bash
source .venv/bin/activate
python run.py
```

---

## Install profiles (extras)

Install only what you need:

Dev (recommended)
```bash
python -m pip install -e ".[dev]"
```

Voice
```bash
python -m pip install -e ".[dev,voice]"
```

Google integrations
```bash
python -m pip install -e ".[dev,google]"
```

API mode
```bash
python -m pip install -e ".[dev,api]"
```

### Notes about constraints / setuptools

> This repo pins setuptools<82 (commonly via constraints.txt) because setuptools 82 removed pkg_resources, and some sdists still expect it during build.

---

## Voice setup (optional)

Voice installs can fail without OS-level audio deps.

### Ubuntu / WSL
```bash
sudo apt update
sudo apt install -y ffmpeg portaudio19-dev libasound2-dev libsndfile1

# then:
python -m pip install -e ".[dev,voice]"
```

## Notes

> On WSL, microphone capture can be OS/driver dependent. If you hit mic issues, verify your audio stack before blaming Python deps.

---

## API mode (optional)
```bash
python -m pip install -e ".[dev,api]"
uvicorn <your_fastapi_module>:app --reload
```

> (Adjust the module path to whatever your repo uses.)

---

# Testing + linting

### Tests
```bash
pytest -q
```

### Ruff
```bash
ruff check . --fix
```

### Black
```
black .
```

### CI-friendly test reports
```bash
mkdir -p reports/pytest

pytest \
  -ra \
  --junitxml=reports/pytest/junit.xml \
  --cov=base \
  --cov-report=term \
  --cov-report=html:reports/pytest/htmlcov \
  --cov-report=xml:reports/pytest/coverage.xml \
  --log-file=reports/pytest/pytest.log \
  --log-level=INFO
```

Note:
> If your coverage import root differs, switch --cov=base to match your actual package import root.

---

# Troubleshooting

---

## Phase 0 monorepo additions (non-destructive)

This repository now also includes a TypeScript monorepo foundation under:

- `apps/web` (Next.js shell)
- `apps/core` (core API)
- `apps/ai` (AI adapter boundary)
- `apps/mcp-gateway` (canonical MCP gateway + capability registry edge surface)
- `packages/*` shared packages (`shared`, `database`, `config`, `security`, etc.)

Use these root workspace commands for the new stack:

```bash
pnpm install
pnpm db:reset
pnpm db:migrate
pnpm db:seed
pnpm lint
pnpm typecheck
pnpm test
```

The existing Aerith/Ultron project structure and workflows remain intact.

### Redesign hardening references (WS12)

For rollout hardening guidance across redesign workstreams, use:

- [`docs/redesign/ws12-rollout-hardening.md`](docs/redesign/ws12-rollout-hardening.md)
- [`docs/local-dev-startup.md`](docs/local-dev-startup.md)
- [`.github/workflows/ci.yml`](.github/workflows/ci.yml)

This WS12 guidance provides:

- redesign validation matrix coverage (WS01–WS12)
- migration rehearsal and local rollout checklists
- rollback/recovery runbook steps
- known limitations and operational boundaries
