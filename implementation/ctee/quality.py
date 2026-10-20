"""Measured SNR guard, calibrated using only calibration devices at 20 dB."""
from pathlib import Path
import json
import numpy as np
from .features import phasors
from .simulator import Circuit,BEHAVIOURS,supported,episode
from .data import save_json

def estimate_snr(waves,frequencies,selected,sample_rate=5000):
    values=[]
    for p in selected:
        w=np.asarray(waves[p],dtype=float)
        _,se=phasors(w,frequencies[p],sample_rate)
        noise=se**2*w.shape[-1]
        total=np.mean(w**2,axis=1)
        signal=np.maximum(total-noise,1e-24)
        values.extend(np.clip(10*np.log10(signal/np.maximum(noise,1e-24)),-50,120).tolist())
    return float(min(values))

def calibrate_guard(root):
    root=Path(root);manifest=json.loads((root/'data/manifest.json').read_text())
    values=[];groups=set()
    for gi,entry in enumerate(manifest['groups']):
        if entry['split']!='calibration':continue
        c=Circuit(**{k:entry[k] for k in Circuit.__dataclass_fields__})
        groups.add(gi)
        for bi,b in enumerate(BEHAVIOURS):
            if not supported(c,b):continue
            w,f=episode(c,b,44501+gi*7+bi,20)
            values.append(estimate_snr(w,f,range(5)))
    threshold=float(np.quantile(values,.01)-3.)
    path=root/'artifacts/calibration.json';cal=json.loads(path.read_text())
    cal['minimum_input_snr_db']=threshold
    cal['input_quality_calibration']={'split':'calibration','physical_groups':len(groups),'episodes':len(values),
       'reference_condition':'20 dB white noise, the lower trained noise level',
       'rule':'1st percentile of minimum measured channel/probe SNR minus a 3 dB margin',
       'threshold_db':threshold,'does_not_guarantee_unknown_domain_detection':True}
    save_json(path,cal);save_json(root/'reports/quality_guard_calibration.json',cal['input_quality_calibration'])
    print('Input-quality guard:',threshold,'dB, from calibration devices only')
    return cal['input_quality_calibration']

