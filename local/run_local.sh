#!/usr/bin/env bash
# Run the local FastAPI (Ollama) app for local testing.
# Uses .env in the repo root. Not deployed to Vercel (see .vercelignore).
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f "venv/bin/activate" ]; then
  echo "Activating venv..."
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

echo "Starting local FastAPI (Ollama) on http://127.0.0.1:8000"
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000