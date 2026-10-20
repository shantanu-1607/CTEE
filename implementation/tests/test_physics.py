import numpy as np
import pytest
from ctee.simulator import *
from ctee.features import extract, N_BASE, physics_logits

@pytest.fixture
def circuit():return sample_circuit('RLC',0,np.random.default_rng(42))

def test_voltage_laws_and_controller_energy(circuit):
    for behaviour,target in [('CC',0),('CP',-1),('CZ',1)]:
        waves,fs=episode(circuit,behaviour)
        x=extract(waves,fs,[0,1,2])
        assert abs(x[N_BASE]-target)<2e-4
        for p in [0,1,2]:
            w,meta=clean_probe(circuit,behaviour,p)
            p_terminal=np.mean(w[0]*w[1]);p_branch=meta['gain']*p_terminal
            p_controller=(meta['gain']-1)*p_terminal
            assert abs(p_terminal+p_controller-p_branch)<1e-6

def test_known_impedances():
    c=Circuit('test','RLC',10.,.1,100e-6,.01,0.,(.001,.002,.001))
    expected=10.01+1j*(2*np.pi*50*.1-1/(2*np.pi*50*100e-6))
    assert abs(impedance(c,50)-expected)<1e-10

def test_unsupported_cp_is_rejected():
    c=sample_circuit('LC',0,np.random.default_rng(0))
    with pytest.raises(ValueError):clean_probe(c,'CP',0)
    assert len(SUPPORT)==18

def test_no_unselected_probe_oracle(circuit):
    w,f=episode(circuit,'CZ');a=extract(w,f,[0,1])
    w[2:]=np.nan;b=extract(w,f,[0,1])
    np.testing.assert_allclose(a,b)

def test_one_voltage_cannot_establish_behaviour(circuit):
    w,f=episode(circuit,'CC');x=extract(w,f,[0])
    assert x[N_BASE+3]==0
    assert np.all(physics_logits(x)[0]==0)

def test_phase_and_spectral_components():
    for label in COMPOSITIONS:
        c=sample_circuit(label,0,np.random.default_rng(9))
        w,f=episode(c,'CZ');x=extract(w,f,[0,3])
        assert x[N_BASE+8]==1
        fractions=x[N_BASE+4:N_BASE+7]
        for j,name in enumerate('RLC'):
            if name in label:assert fractions[j]>.02
            else:assert fractions[j]<.005

def test_startup_solver_matches_settled_phasor(circuit):
    t,v,i=transient(circuit,duration=1.)
    ix=t>.8
    z=impedance(circuit,50)
    expected=np.imag(np.exp(2j*np.pi*50*t)/z)
    assert np.sqrt(np.mean((i[ix]-expected[ix])**2))<.002*max(np.sqrt(np.mean(expected[ix]**2)),1e-8)

def test_noise_snr():
    c=sample_circuit('R',0,np.random.default_rng(5));w,_=clean_probe(c,'CZ',0)
    y=corrupt(w,np.random.default_rng(99),20)
    snr=10*np.log10(np.mean(w**2,axis=1)/np.mean((y-w)**2,axis=1))
    np.testing.assert_allclose(snr,20,atol=.001)

def test_nonfinite_selected_measurement_rejected(circuit):
    w,f=episode(circuit,'CZ');w[0,0,3]=np.nan
    with pytest.raises(ValueError):extract(w,f,[0])
