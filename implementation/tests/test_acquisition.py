"""Operational stopping must never acquire diagnostic future observations."""
import numpy as np
import pytest
from ctee.inference import trace_policy
from ctee.simulator import sample_circuit,episode
from ctee.quality import estimate_snr

class ProbeRecorder:
    def __init__(self,quality=True):self.calls=[];self.quality=quality
    def predict(self,waves,frequencies,selected,startup):
        self.calls.append(selected.copy())
        return {'accepted':len(selected)>=3,'input_quality_passed':self.quality,
                'evidence':{'slope_valid':len(selected)>1,'spectral_valid':3 in selected},
                'behaviour_probabilities':[.9,.05,.05],
                'composition_probabilities':[.9,.02,.02,.02,.02,.01,.01]}

def test_operational_policy_stops_after_acceptance():
    predictor=ProbeRecorder()
    result=trace_policy(predictor,None,None)
    assert predictor.calls==[[0],[0,1],[0,1,3]]
    assert result['decision']['budget']==3 and len(result['trace'])==3

def test_low_quality_stops_without_reading_future_probes():
    predictor=ProbeRecorder(False)
    result=trace_policy(predictor,None,None)
    assert predictor.calls==[[0]] and result['status']=='inconclusive'

def test_diagnostics_preserve_decision_but_continue_for_budget_curves():
    predictor=ProbeRecorder()
    result=trace_policy(predictor,None,None,diagnostic=True)
    assert len(predictor.calls)==5 and result['decision']['budget']==3
def test_quality_estimate_separates_training_noise_and_severe_noise():
    c=sample_circuit('RLC',0,np.random.default_rng(80))
    good,f=episode(c,'CZ',7,20)
    bad,_=episode(c,'CZ',8,10,'coloured')
    assert 18<estimate_snr(good,f,[0,1,3])<22
    assert estimate_snr(bad,f,[0,1,3])<15

def test_quality_guard_does_not_read_unchosen_probes():
    c=sample_circuit('RL',0,np.random.default_rng(31));w,f=episode(c,'CZ',1,20)
    original=estimate_snr(w,f,[0,1]);w[2:]=np.nan
    assert original==estimate_snr(w,f,[0,1])

def test_runtime_rejects_signal_below_calibrated_range():
    from ctee.runtime import DEFAULT_RUN
    root=DEFAULT_RUN
    if not (root/'artifacts/calibration.json').exists():pytest.skip('Train first')
    from ctee.inference import Predictor
    predictor=Predictor(root)
    if 'minimum_input_snr_db' not in predictor.calibration:pytest.skip('Calibrate quality guard first')
    c=sample_circuit('RLC',0,np.random.default_rng(99));w,f=episode(c,'CZ',5,10,'coloured')
    result=predictor.predict(w,f,[0,1,3])
    assert not result['accepted'] and not result['input_quality_passed']
    assert any('cleaner signal' in reason for reason in result['reasons'])


def test_full_measurement_returns_the_five_probe_decision():
    predictor=ProbeRecorder()
    result=trace_policy(predictor,None,None,policy='full')
    assert len(predictor.calls)==5
    assert result['decision']['budget']==5
