#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."

echo "== Aerith/Ultron bootstrap =="

# 1) Ensure python3 exists
if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is not installed."
  echo "Ubuntu/WSL: sudo apt update && sudo apt install -y python3 python3-venv python3-pip"
  exit 1
fi

# 2) Show version
PYV="$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
echo "python3 version: ${PYV}"

# 2.5) Ensure ensurepip exists (Debian/Ubuntu splits it into python3-venv / python3.X-venv)
if ! python3 -c "import ensurepip" >/dev/null 2>&1; then
  echo "ERROR: ensurepip is not available (can't seed pip inside venv)."
  echo "Ubuntu/WSL: sudo apt update && sudo apt install -y python3-venv"
  echo "If that doesn't work, install the versioned package (example): sudo apt install -y python3.12-venv"
  exit 1
fi

# 3) Ensure venv module works (common missing package on Ubuntu)
if ! python3 -m venv --help >/dev/null 2>&1; then
  echo "ERROR: python3 venv module is missing."
  echo "Ubuntu/WSL: sudo apt update && sudo apt install -y python3-venv"
  exit 1
fi

# 4) Create venv
rm -rf .venv
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

# 5) Upgrade pip tooling + install dev deps
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"

# 6) Create .env if sample exists
if [[ -f .env.sample && ! -f .env ]]; then
  cp .env.sample .env
  echo "Created .env from .env.sample"
fi

# 7) Sanity check
pytest -q

echo "✅ Bootstrap complete."
echo "Next: source .venv/bin/activate && python run.py"