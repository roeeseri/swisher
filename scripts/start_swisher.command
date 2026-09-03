#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"
if [ ! -x .venv/bin/python ]; then
  echo "Swisher is not installed yet. Run: python3 -m venv .venv"
  read -r
  exit 1
fi
.venv/bin/python -m streamlit run app.py
