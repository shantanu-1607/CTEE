from pathlib import Path
import pytest

def test_local_dashboard_loads_and_classifies():
    root=Path(__file__).resolve().parents[1]
    from ctee.runtime import DEFAULT_RUN
    if not (DEFAULT_RUN/'artifacts/selected_model.npz').exists():pytest.skip('Train first')
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(str(root/'app.py'),default_timeout=30).run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
    assert len(app.metric)==4
    assert app.metric[0].value in ['CC','CP','CZ']

    # A single voltage observation cannot identify the response law.
    app.radio[0].set_value('Held-out benchmark example').run()
    app.slider[0].set_value(1).run()
    assert not app.exception
    assert app.metric[3].value=='Inconclusive'


def test_parameter_lab_previews_a_bias_edit():
    from ctee.runtime import DEFAULT_RUN
    if not (DEFAULT_RUN/'artifacts/selected_model.npz').exists():pytest.skip('Train first')
    from streamlit.testing.v1 import AppTest
    import json
    root=Path(__file__).resolve().parents[1]
    app=AppTest.from_file(str(root/'app.py'),default_timeout=30).run()
    app.button[0].click().run()
    next(item for item in app.selectbox if item.label=='Weight or bias tensor').set_value('b_composition').run()
    next(item for item in app.slider if item.label=='Temporary weight change').set_value(.5).run()
    next(item for item in app.button if item.label=='Preview this weight change').click().run()
    assert not app.exception
    result=json.loads(app.json[-1].value)
    assert result['after']-result['before']==pytest.approx(.5)
    assert result['edited']['composition'][0]>result['original']['composition'][0]
