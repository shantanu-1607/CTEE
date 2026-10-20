from pathlib import Path
import pytest

def test_local_dashboard_loads_and_classifies():
    root=Path(__file__).resolve().parents[1]
    if not (root/'artifacts/selected_model.pt').exists():pytest.skip('Train first')
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(str(root/'app.py'),default_timeout=30).run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
    assert len(app.metric)==4
    assert app.metric[0].value in ['CC','CP','CZ']
