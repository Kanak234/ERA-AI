#!/usr/bin/env bash
# ERA AI — one-command startup (spec Section 22)
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo ""
echo "  ERA AI — starting up"
echo "  root: $ROOT"
echo ""

# 1. Python
if ! command -v python3 >/dev/null 2>&1; then
  echo "  ✖ python3 not found. Install Python 3.10+."
  exit 1
fi

# 2. venv + deps
if [ ! -d ".venv" ]; then
  echo "  • creating virtual environment (.venv)"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "  • installing dependencies"
pip install -q --upgrade pip
pip install -q -r requirements.txt

# 3. Ollama check (non-fatal)
if command -v ollama >/dev/null 2>&1; then
  if ! curl -s http://localhost:11434/api/version >/dev/null 2>&1; then
    echo "  • starting ollama serve in background"
    (ollama serve >/dev/null 2>&1 &) || true
    sleep 2
  fi
else
  echo "  ⚠ ollama not found — install from https://ollama.com (GUI will still open)"
fi

# 4. Health check (Phase 0 gate)
python scripts/health_check.py || true

# 5. Launch
echo "  • launching ERA AI"
python run_era.py
