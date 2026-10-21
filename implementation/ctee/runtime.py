"""Run isolation keeps new results separate from the historical library prototype."""
from pathlib import Path
import json
IMPLEMENTATION_ROOT=Path(__file__).resolve().parents[1]
DEFAULT_RUN=IMPLEMENTATION_ROOT/'runs/numpy_v1'
DEFAULT_CONFIG=IMPLEMENTATION_ROOT/'configs/experiments.json'

def settings(path=None):
    return json.loads(Path(path or DEFAULT_CONFIG).read_text())
