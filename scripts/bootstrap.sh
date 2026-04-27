#!/usr/bin/env bash

main() {
  set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "== Aerith bootstrap =="

# Refuse to run inside an active venv (script deletes .venv)
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  echo "ERROR: You are currently inside a virtualenv: $VIRTUAL_ENV"
  echo "Open a fresh shell (or run 'deactivate') and re-run:"
  echo "  ./scripts/bootstrap.sh"
  return 1
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
  return 1
fi

echo "Using: $PYTHON_BIN"
"$PYTHON_BIN" --version || return 1

# Ensure venv + ensurepip are present
if ! "$PYTHON_BIN" -c "import venv, ensurepip" >/dev/null 2>&1; then
  echo "ERROR: venv/ensurepip missing."
  echo "Ubuntu/WSL: sudo apt update && sudo apt install -y python3-venv"
  echo "If needed: sudo apt install -y python3.12-venv or python3.11-venv"
  return 1
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
"$PYTHON_BIN" -m venv .venv || return 1
# shellcheck disable=SC1091
source .venv/bin/activate || return 1

# Make constraints apply everywhere (including build isolation)
export PIP_CONSTRAINT="$CONSTRAINTS_FILE"

python --version || return 1

# Upgrade installer tooling (honor constraints)
python -m pip install --upgrade pip wheel || return 1
python -m pip install --upgrade "setuptools<82" || return 1

# Install profile:
#   default: dev (safe)
#   set AERITH_EXTRAS=dev,voice to include voice deps
AERITH_EXTRAS="${AERITH_EXTRAS:-dev,voice}"
echo "Installing extras: ${AERITH_EXTRAS}"


# If voice extras are requested, install required system deps (Ubuntu/Debian/WSL)
# Can be disabled with: INSTALL_SYSTEM_DEPS=0 ./scripts/bootstrap.sh
INSTALL_SYSTEM_DEPS="${INSTALL_SYSTEM_DEPS:-1}"

if [[ "$INSTALL_SYSTEM_DEPS" == "1" ]]; then
  if [[ ",${AERITH_EXTRAS}," == *",voice,"* ]]; then
    if command -v apt-get >/dev/null 2>&1; then
      echo "Installing system deps for voice (apt-get)..."
      sudo apt-get update || return 1

      apt_packages=(
        build-essential
        python3-dev
        libasound2-dev
        portaudio19-dev
        libsndfile1
        ffmpeg
      )

      sudo apt-get install -y "${apt_packages[@]}" || return 1
    fi
  fi
fi

python -m pip install -e ".[${AERITH_EXTRAS}]" || return 1

if command -v corepack >/dev/null 2>&1; then
  corepack enable
  corepack prepare pnpm@9.12.3 --activate || return 1
fi

if command -v pnpm >/dev/null 2>&1; then
  pnpm install --frozen-lockfile || return 1
else
  echo "WARNING: pnpm is not available; TS/JS workspace tooling was not installed."
fi

# Create .env if sample exists
if [[ -f .env.sample && ! -f .env ]]; then
  cp .env.sample .env || return 1
  echo "Created .env from .env.sample"
fi

# Sanity check (don't hide failures)
pytest -q || return 1

echo "✅ Bootstrap complete."
echo "Next:"
echo "  source .venv/bin/activate"
echo "  python run.py"

}

if ! main "$@"; then
  echo
  echo "❌ Bootstrap stopped before completion."
  echo "Fix the error above, then re-run:"
  echo "  ./scripts/bootstrap.sh"
fi
