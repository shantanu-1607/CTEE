#!/bin/zsh
set -eu
cd -- "$(dirname "$0")"
.venv/bin/python -m ctee all --configurations 100 --epochs 100
.venv/bin/python -m pytest -q

