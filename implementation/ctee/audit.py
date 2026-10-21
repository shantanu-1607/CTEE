"""Independent analytical/ODE/measurement checks before data generation."""
from pathlib import Path
import numpy as np
from .simulator import *
from .features import extract,N_BASE
from .data import save_json

def run(root):
    rng=np.random.default_rng(10);checks=[];maximum_slope_error=0.;maximum_control_error=0.
    for c in COMPOSITIONS:
        fixture=sample_circuit(c,0,rng)
        for b in BEHAVIOURS:
            if not supported(fixture,b):continue
            w,f=episode(fixture,b);x=extract(w,f,[0,1,2]);target={'CC':0,'CP':-1,'CZ':1}[b]
            slope_error=abs(float(x[N_BASE])-target);maximum_slope_error=max(maximum_slope_error,slope_error)
            i0=abs(1/impedance(fixture,50));p0=(1/impedance(fixture,50)).real
            for p in range(5):
                v,i=w[p];power=np.mean(v*i);irms=np.sqrt(np.mean(i*i))
                error=(abs(irms-i0)/i0 if b=='CC' else abs(power-p0)/p0 if b=='CP' else 0.)
                maximum_control_error=max(maximum_control_error,float(error))
            checks.append({'composition':c,'behaviour':b,'current_slope':float(x[N_BASE]),'slope_error':slope_error})
    assert maximum_slope_error<.001
    assert maximum_control_error<.001
    result={'status':'passed','supported_pairs_tested':len(checks),'maximum_slope_error':maximum_slope_error,
            'maximum_relative_regulation_error':maximum_control_error,'measurements':checks,
            'independent_ngspice_check':'not performed; analytical and manual state-equation checks only',
            'scope':'ideal controlled-source simulation; no hardware validation'}
    save_json(Path(root)/'reports/phase1_physics_audit.json',result)
    print(result)
    return result
