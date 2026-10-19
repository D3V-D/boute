#!/usr/bin/env bash
set -e
cd -- "$(dirname -- "$0")"
if ! command -v python3 >/dev/null 2>&1; then
  echo 'Install Python 3.10 or newer, python3-tk, and python3-venv using your Linux package manager.'
  read -r -p 'Press Enter to close…'
  exit 1
fi
python3 bootstrap.py
