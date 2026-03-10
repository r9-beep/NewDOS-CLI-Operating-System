#!/usr/bin/env bash
# Launch the NewDOS GUI launcher.
# Requires Python 3 with tkinter (python3-tk).
set -e
cd "$(dirname "$0")"
python3 newdos_launcher.py "$@"
