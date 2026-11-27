import pandas as pd
import pytest

from load_classifier.data import MEASUREMENTS


@pytest.fixture
def windows():
    rows = []
    for i, start in enumerate((0, 30000, 60000, 120000, 240000)):
        time = start / 30000
        row = {name: 2 * time + 3 for name in MEASUREMENTS}
        row.update(id=1, window_idx=i, start_sample=start, end_sample=start + 499,
                   t_start_s=time, t_end_s=(start + 500) / 30000)
        rows.append(row)
    return pd.DataFrame(rows)
