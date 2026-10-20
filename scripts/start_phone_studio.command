#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH=src
exec .venv/bin/python -m swisher.studio.server --port 8770 --phone
