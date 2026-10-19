#!/bin/bash
set -e
cd -- "$(dirname -- "$0")"
if ! command -v python3 >/dev/null 2>&1; then
  read -r -p 'Python is required. Open the Python installer download page? [y/N] ' answer
  if [[ "$answer" =~ ^[Yy]$ ]]; then open 'https://www.python.org/downloads/macos/'; fi
  echo 'Install Python 3.10 or newer, then open this launcher again.'
  read -r -p 'Press Enter to close…'
  exit 1
fi
python3 bootstrap.py
