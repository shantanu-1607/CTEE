"""Fresh configuration pool to confirm the frozen classifier and quality guard.

Added after the first stress test exposed overconfidence in coloured noise.
No weights, feature choices, or thresholds are fitted on this confirmation pool.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from .simulator import *
from .features import extract
from .inference import Predictor
from .model import load_model,predict_logits,probabilities
from .metrics import score,group_bootstrap_difference
from .data import save_json

def confirm(root,per_topology=20,seed=20261021):
    root=Path(root);rng=np.random.default_rng(seed);predictor=Predictor(root)
    baseline,_=load_model(root/'artifacts/baseline_robust_11.pt')
    rows=[];devices=[]
    for ci,composition in enumerate(COMPOSITIONS):
        for index in range(per_topology):
            circuit=sample_circuit(composition,200000+index,rng);group=len(devices);devices.append(circuit.to_dict())
            _,v,i=transient(circuit,sample_rate=5000);base_start=np.stack([v,i])*np.sqrt(2)
            for bi,b in enumerate(BEHAVIOURS):
                if not supported(circuit,b):continue
                start=base_start.copy();start[1]*=clean_probe(circuit,b,0)[1]['gain']
                for condition,snr,kind in [('clean',None,'nominal'),('noise20',20,'nominal'),('coloured10',10,'coloured')]:
                    simseed=int(rng.integers(2**30));w,f=episode(circuit,b,simseed,snr,kind)
                    ss=corrupt(start,np.random.default_rng(simseed+1),snr,kind)
                    r=predictor.predict(w,f,list(range(5)),ss)
                    x=extract(w,f,startup=ss);lb,lc=predict_logits(baseline,x[None,:])
                    rows.append({'group':group,'condition':condition,'yb':bi,'yc':ci,
                          'pb':r['behaviour_probabilities'],'pc':r['composition_probabilities'],
                          'bb':probabilities(lb)[0].tolist(),'bc':probabilities(lc)[0].tolist(),
                          'accepted':r['accepted'],'quality_db':r['estimated_minimum_snr_db']})
    results={}
    for condition in ['clean','noise20','coloured10']:
        rr=[r for r in rows if r['condition']==condition];yb=np.array([r['yb'] for r in rr]);yc=np.array([r['yc'] for r in rr])
        pb=np.array([r['pb'] for r in rr]);pc=np.array([r['pc'] for r in rr]);bb=np.array([r['bb'] for r in rr]);bc=np.array([r['bc'] for r in rr])
        accepted=np.array([r['accepted'] for r in rr]);correct=(pb.argmax(1)==yb)&(pc.argmax(1)==yc)
        base=(bb.argmax(1)==yb)&(bc.argmax(1)==yc)
        results[condition]={'enhanced':score(yb,yc,pb,pc),'baseline':score(yb,yc,bb,bc),
            'coverage':float(accepted.mean()),'accepted_joint_accuracy':float(correct[accepted].mean()) if accepted.any() else None,
            'group_comparison':group_bootstrap_difference(np.array([r['group'] for r in rr]),correct.astype(float),base.astype(float))}
    record={'seed':seed,'physical_configurations':len(devices),'episodes_per_condition':len(rows)//3,
            'model_sha256':hashlib.sha256((root/'artifacts/selected_model.pt').read_bytes()).hexdigest(),
            'method':'Fresh independent simulator configurations. Classifier frozen before the first test; quality guard calibrated solely on original calibration devices. Added after the initial coloured-noise failure was observed.',
            'results':results,'devices':devices,'predictions':rows}
    save_json(root/'reports/fresh_confirmation.json',record)
    print('Fresh confirmation:',{c:{'joint':r['enhanced']['joint_accuracy'],'coverage':r['coverage'],'accepted_accuracy':r['accepted_joint_accuracy']} for c,r in results.items()})
    return record
