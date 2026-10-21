#!/bin/zsh
set -eu
cd -- "$(dirname "$0")"
exec .venv_numpy/bin/python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false

