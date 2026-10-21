#!/bin/zsh
set -eu
cd -- "$(dirname "$0")"
.venv_numpy/bin/python -m ctee all
.venv_numpy/bin/python -m pytest -q

