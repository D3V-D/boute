"""Dependency-aware, shared launcher. Run with Python 3.10 or newer."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import venv


def main():
    if sys.version_info < (3, 10):
        raise RuntimeError('Boute requires Python 3.10 or newer. Install it from python.org.')
    root = Path(__file__).resolve().parent
    os.chdir(root)
    environment = root / '.venv'
    python = environment / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not python.exists():
        print('Preparing Boute…')
        venv.EnvBuilder(with_pip=True).create(environment)
    requirements = root / 'requirements.txt'
    digest = hashlib.sha256(requirements.read_bytes()).hexdigest()
    stamp = environment / '.boute-requirements'
    if not stamp.exists() or stamp.read_text() != digest:
        subprocess.run([str(python), '-m', 'pip', 'install', '-r', str(requirements)], check=True)
        stamp.write_text(digest)
    check = subprocess.run([str(python), '-c', 'import tkinter, tkinterdnd2, customtkinter, cv2, numpy, PIL'])
    if check.returncode:
        raise RuntimeError('A dependency is missing. On Linux install python3-tk and python3-venv using your system package manager. Delete .venv to repair other dependency problems.')
    subprocess.run([str(python), '-m', 'boute'], check=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'\nBoute could not start: {error}', file=sys.stderr)
        input('Press Enter to close…')
        sys.exit(1)
