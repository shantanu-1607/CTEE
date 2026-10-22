from pathlib import Path
import json
import numpy as np
from .features import extract,evidence_summary,N_BASE
from .model import load_model,predict_logits,probabilities
from .simulator import PROBES,DURATION
from .quality import estimate_snr

class Predictor:
    def __init__(self,root):
        root=Path(root)
        self.model,_=load_model(root/'artifacts/selected_model.npz')
        self.calibration=json.loads((root/'artifacts/calibration.json').read_text())

    def predict_features(self,x):
        b,c=predict_logits(self.model,np.atleast_2d(x))
        budget=int(np.sum(np.asarray(x)[13:N_BASE:14]>.5))
        cal=self.calibration.get('per_budget',{}).get(str(budget),self.calibration)
        pb=probabilities(b,cal['behaviour_temperature'])[0]
        pc=probabilities(c,cal['composition_temperature'])[0]
        evidence=evidence_summary(x)
        confidence=float(min(pb.max(),pc.max()))
        reasons=[]
        if not evidence['slope_valid']:reasons.append('Need distinct settled voltage levels')
        if not evidence['spectral_valid']:reasons.append('Need stronger frequency/harmonic excitation')
        if evidence['spectral_fit_error']>=.2:reasons.append('Measured impedance poorly fits the declared series family')
        if confidence<cal['threshold']:reasons.append('Confidence is below the calibration threshold')
        gates=self.model.physics_gate_values(np.atleast_2d(x)[:,:self.model.config.input_dim])
        return {'physics_gates':{k:float(v[0]) for k,v in gates.items()},'features':np.asarray(x).tolist(),
                'behaviour_probabilities':pb.tolist(),'composition_probabilities':pc.tolist(),
                'behaviour':int(pb.argmax()),'composition':int(pc.argmax()),'confidence':confidence,
                'accepted':not reasons,'reasons':reasons,'evidence':evidence}

    def predict(self,waves,frequencies,selected,startup=None,sample_rate=5000):
        result=self.predict_features(extract(waves,frequencies,selected,sample_rate,startup))
        snr=estimate_snr(waves,frequencies,selected,sample_rate)
        result['estimated_minimum_snr_db']=snr
        limit=self.calibration.get('minimum_input_snr_db',-50.)
        result['input_quality_passed']=snr>=limit
        if snr<limit:
            result['accepted']=False
            result['reasons'].append(f'Measurement quality ({snr:.1f} dB estimated SNR) is below the calibrated operating range; record a cleaner signal')
        return result

def next_probe(selected,result):
    """No access to the unselected waveforms or any ground truth."""
    selected=set(selected)
    if not result['evidence']['slope_valid']:
        for p in [1,2]:
            if p not in selected:return p
    if not result['evidence']['spectral_valid'] and 3 not in selected:return 3
    b=max(result['behaviour_probabilities']);c=max(result['composition_probabilities'])
    order=[2,3,4,1] if b<c else [3,4,2,1]
    return next((p for p in order if p not in selected),None)

def trace_policy(predictor,waves,frequencies,startup=None,policy='adaptive',seed=0,diagnostic=False):
    """Stop operational acquisition; only offline diagnostics request future probes."""
    if policy not in {'full','fixed','random','adaptive'}:raise ValueError('Unknown acquisition policy')
    selected=[0];trace=[]
    rng=np.random.default_rng(seed)
    order=[0,1,2,3,4] if policy in {'fixed','full'} else [0,*rng.permutation([1,2,3,4]).tolist()]
    for _ in range(5):
        r=predictor.predict(waves,frequencies,selected,startup)
        r['selected']=selected.copy();r['budget']=len(selected)
        r['recording_seconds_lower_bound']=.3+len(selected)*DURATION
        trace.append(r)
        stop=(r['accepted'] and policy!='full') or (policy=='adaptive' and not r.get('input_quality_passed',True))
        if len(selected)==5 or (stop and not diagnostic):break
        p=next_probe(selected,r) if policy=='adaptive' else order[len(selected)]
        selected.append(p)
    accepted=trace[-1] if policy=='full' else next((r for r in trace if r['accepted']),trace[-1])
    if policy=='adaptive' and not trace[0].get('input_quality_passed',True):
        # Continue computing diagnostic budget curves, but stop the actual acquisition.
        accepted=trace[0]
    return {'trace':trace,'decision':accepted,'policy':policy,
            'status':'accepted' if accepted['accepted'] else 'inconclusive'}

