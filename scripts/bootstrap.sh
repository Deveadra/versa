#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "== Aerith bootstrap =="

# Refuse to run inside an active venv (script deletes .venv)
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  echo "ERROR: You are currently inside a virtualenv: $VIRTUAL_ENV"
  echo "Open a fresh shell (or run 'deactivate') and re-run:"
  echo "  ./scripts/bootstrap.sh"
  exit 1
fi

# Pick a Python (allow override)
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  for c in python3.11 python3.12 python3; do
    if command -v "$c" >/dev/null 2>&1; then
      PYTHON_BIN="$(command -v "$c")"
      break
    fi
  done
fi

if [[ -z "$PYTHON_BIN" ]]; then
  echo "ERROR: python3 not found."
  echo "Ubuntu/WSL: sudo apt update && sudo apt install -y python3 python3-venv python3-pip"
  exit 1
fi

echo "Using: $PYTHON_BIN"
"$PYTHON_BIN" --version

# Ensure venv + ensurepip are present
if ! "$PYTHON_BIN" -c "import venv, ensurepip" >/dev/null 2>&1; then
  echo "ERROR: venv/ensurepip missing."
  echo "Ubuntu/WSL: sudo apt update && sudo apt install -y python3-venv"
  echo "If needed: sudo apt install -y python3.12-venv or python3.11-venv"
  exit 1
fi

# Constraints (guardrail for setuptools>=82 removing pkg_resources)
CONSTRAINTS_FILE="${CONSTRAINTS_FILE:-$ROOT/constraints.txt}"
if [[ ! -f "$CONSTRAINTS_FILE" ]]; then
  cat > "$CONSTRAINTS_FILE" <<'EOF'
setuptools<82
EOF
fi

# Create venv cleanly
rm -rf .venv
"$PYTHON_BIN" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

# Make constraints apply everywhere (including build isolation)
export PIP_CONSTRAINT="$CONSTRAINTS_FILE"

python --version

# Upgrade installer tooling (honor constraints)
python -m pip install --upgrade pip wheel
python -m pip install --upgrade "setuptools<82"

# Install profile:
#   default: dev (safe)
#   set AERITH_EXTRAS=dev,voice to include voice deps
AERITH_EXTRAS="${AERITH_EXTRAS:-dev}"
echo "Installing extras: ${AERITH_EXTRAS}"
python -m pip install -e ".[${AERITH_EXTRAS}]"

# Create .env if sample exists
if [[ -f .env.sample && ! -f .env ]]; then
  cp .env.sample .env
  echo "Created .env from .env.sample"
fi

# Sanity check (don't hide failures)
pytest -q

echo "✅ Bootstrap complete."
echo "Next:"
echo "  source .venv/bin/activate"
echo "  python run.py"