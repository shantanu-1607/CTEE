"""Frozen-weight counterfactual: replace learned gates with exactly one half."""
import copy
import numpy as np
from .model import predict_logits,probabilities
from .metrics import score,group_bootstrap_difference


def analyze_gates(model,x,yb,yc,conditions,groups):
    if not model.config.learned_physics_gate:raise ValueError('Counterfactual requires learned gates')
    fixed=copy.deepcopy(model)
    for head in ['behaviour','composition']:
        fixed.params[f'W_gate_{head}'][:]=0.;fixed.params[f'b_gate_{head}'][:]=0.
    gates=model.physics_gate_values(x)
    outputs=[]
    for candidate in [model,fixed]:
        b,c=predict_logits(candidate,x);outputs.append((probabilities(b),probabilities(c)))
    results={}
    for condition in np.unique(conditions):
        mask=conditions==condition;learned={};correct=[]
        for label,(pb,pc) in zip(['learned','constant_half'],outputs):
            learned[label]=score(yb[mask],yc[mask],pb[mask],pc[mask])
            correct.append(((pb.argmax(1)==yb)&(pc.argmax(1)==yc))[mask].astype(float))
        learned['paired_group_comparison']=group_bootstrap_difference(groups[mask],*correct)
        learned['gates']={head:{'mean':float(values[mask].mean()),'sd':float(values[mask].std()),
                         'p10':float(np.quantile(values[mask],.1)),'p90':float(np.quantile(values[mask],.9))}
                         for head,values in gates.items()}
        results[str(condition)]=learned
    return {'conditions':results,'method':'Same frozen neural weights; learned sigmoid evidence gates versus constant 0.5 gates. No retraining or test-set selection.',
            'caution':'A counterfactual sensitivity check complements the separately trained fixed-prior ablation; it is not a new trained baseline.'}
