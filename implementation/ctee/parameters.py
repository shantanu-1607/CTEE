"""Inspect learned tensors and test isolated weight edits without altering deployment."""
import copy,csv,io
import numpy as np
from .scratch import TwoHeadMLP
from .model import predict_logits,probabilities


def initial_parameters(model):
    initial=TwoHeadMLP(model.config)
    initial.mean=model.mean.copy();initial.scale=model.scale.copy()
    return initial


def parameter_summary(model):
    initial=initial_parameters(model);rows=[]
    for name,p in model.params.items():
        change=p-initial.params[name]
        rows.append({'tensor':name,'shape':' × '.join(map(str,p.shape)),'count':int(p.size),
            'minimum':float(p.min()),'maximum':float(p.max()),'standard_deviation':float(p.std()),
            'l2_norm':float(np.sqrt(np.sum(p**2))),'change_from_initial_l2':float(np.sqrt(np.sum(change**2))),
            'changed_values':int(np.sum(abs(change)>1e-12))})
    return rows


def export_parameters(model):
    initial=initial_parameters(model);stream=io.StringIO();writer=csv.writer(stream)
    writer.writerow(['tensor','index','initial_value','learned_value','change'])
    for name,p in model.params.items():
        for index in np.ndindex(p.shape):
            writer.writerow([name,':'.join(map(str,index)),float(initial.params[name][index]),float(p[index]),
                             float(p[index]-initial.params[name][index])])
    return stream.getvalue()


def preview_edit(model,x,tensor,index,delta):
    """Return a separate model and its predictions; the deployed object is read-only."""
    if tensor not in model.params:raise ValueError('Unknown parameter tensor')
    shape=model.params[tensor].shape
    if len(index)!=len(shape) or any(not isinstance(i,int) or not 0<=i<n for i,n in zip(index,shape)):
        raise ValueError('Coordinate outside parameter shape')
    if not np.isfinite(delta):raise ValueError('Parameter delta must be finite')
    experimental=copy.deepcopy(model);before=float(model.params[tensor][index])
    experimental.params[tensor][index]+=delta
    predictions=[]
    for candidate in [model,experimental]:
        b,c=predict_logits(candidate,np.atleast_2d(x))
        predictions.append({'behaviour':probabilities(b)[0].tolist(),'composition':probabilities(c)[0].tolist()})
    return experimental,{'tensor':tensor,'index':list(index),'before':before,'after':before+delta,
                         'original':predictions[0],'edited':predictions[1],
                         'scope':'raw probabilities; a manual edit invalidates the original calibration and benchmark results'}
