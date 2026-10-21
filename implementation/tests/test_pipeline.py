from pathlib import Path
import numpy as np
import pytest
from ctee.data import split_indices,load
from ctee.simulator import sample_circuit,episode
from ctee.features import extract,N_FEATURES
from ctee.csv_io import parse_csv,export_csv
from ctee.model import SharedClassifier,save_model,load_model,predict_logits
from ctee.inference import next_probe

def test_parent_split_is_disjoint_and_complete():
    split=split_indices(100,np.random.default_rng(1))
    assert len(split)==100
    assert [list(split.values()).count(x) for x in ['train','validation','calibration','test']]==[60,15,10,15]

def test_two_head_serialization_roundtrip(tmp_path):
    model=SharedClassifier(True,.5)
    x=np.zeros((4,N_FEATURES),dtype=np.float32)
    before=predict_logits(model,x)
    save_model(tmp_path/'model.npz',model,11,'test')
    restored,_=load_model(tmp_path/'model.npz');after=predict_logits(restored,x)
    assert before[0].shape==(4,3) and before[1].shape==(4,7)
    np.testing.assert_allclose(before[0],after[0]);np.testing.assert_allclose(before[1],after[1])

def test_csv_roundtrip_and_missing_probe_mask():
    c=sample_circuit('RL',0,np.random.default_rng(12));w,f=episode(c,'CP')
    text=export_csv(w,f,[0,1,3]);restored,freq,selected,sr=parse_csv(text)
    assert selected==[0,1,3] and abs(sr-5000)<1e-6
    np.testing.assert_allclose(extract(w,f,selected),extract(restored,freq,selected,sr),atol=1e-5)

def test_csv_rejects_time_reversal():
    c=sample_circuit('R',0,np.random.default_rng(4));w,f=episode(c,'CZ')
    text=export_csv(w,f,[0]).splitlines();text[2]=text[1]
    with pytest.raises(ValueError,match='increasing'):parse_csv('\n'.join(text))

def test_policy_requests_voltage_without_hidden_state():
    result={'evidence':{'slope_valid':False,'spectral_valid':False},'behaviour_probabilities':[.9,.05,.05],'composition_probabilities':[1,0,0,0,0,0,0]}
    assert next_probe([0],result)==1

def test_generated_data_has_no_cross_split_parent():
    root=Path(__file__).resolve().parents[1]
    if not (root/'data/benchmark.npz').exists():pytest.skip('Run the data phase first')
    d=load(root)
    groups=[set(d['group'][d['split']==s]) for s in ['train','validation','calibration','test']]
    for i,a in enumerate(groups):
        for b in groups[i+1:]:assert not a&b
    assert np.isfinite(d['X']).all()



def test_manual_temperature_calibration_never_increases_calibration_nll():
    from ctee.model import temperatures, probabilities
    from ctee.metrics import probability_metrics
    z=np.array([[10.,0.],[10.,0.],[0.,10.],[0.,10.]])
    y=np.array([0,1,0,1])
    t=temperatures(z,y)
    assert t>1
    assert probability_metrics(y,probabilities(z,t))['nll']<=probability_metrics(y,probabilities(z))['nll']
