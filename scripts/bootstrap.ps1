
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location (Join-Path $PSScriptRoot "..")

Write-Host "== Aerith/Ultron bootstrap =="

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
  throw "ERROR: Python launcher 'py' not found. Install Python 3.11+ from python.org."
}

Remove-Item -Recurse -Force .venv -ErrorAction SilentlyContinue

py -3 -m venv .venv
& .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"

if (Test-Path .env.sample -and -not (Test-Path .env)) {
  Copy-Item .env.sample .env
  Write-Host "Created .env from .env.sample"
}

pytest -q

Write-Host "✅ Bootstrap complete. Next: python run.py"