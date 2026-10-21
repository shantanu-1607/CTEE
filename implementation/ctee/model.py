"""Electrical-model adapters and explicit one-dimensional probability calibration."""
import json
import numpy as np
from .features import N_BASE,N_FEATURES,FEATURE_NAMES,physics_logits
from .scratch import NetworkConfig,TwoHeadMLP,softmax


def SharedClassifier(enhanced=True,physics_weight=0.,seed=11):
    return TwoHeadMLP(NetworkConfig(N_FEATURES if enhanced else N_BASE,prior_strength=physics_weight,seed=seed))


def save_model(path,model,seed,variant):
    model.save(path,{'seed':int(seed),'variant':variant,'feature_names':FEATURE_NAMES[:model.config.input_dim],
                    'implementation':'manual_numpy','behaviours':['CC','CP','CZ'],'compositions':['R','L','C','RL','RC','LC','RLC']})


def load_model(path):
    model=TwoHeadMLP.load(path)
    with np.load(path,allow_pickle=False) as saved:metadata=json.loads(saved['metadata'].item())['training']
    if metadata.get('feature_names')!=FEATURE_NAMES[:model.config.input_dim]:raise ValueError('Feature schema differs from this checkpoint')
    return model,metadata


def predict_logits(model,x,batch_size=4096):
    x=np.asarray(x,dtype=float)
    if x.ndim!=2 or not len(x) or batch_size<1:raise ValueError('Expected a nonempty feature matrix and positive batch size')
    b=[];c=[]
    for start in range(0,len(x),batch_size):
        raw=x[start:start+batch_size]
        priors=None
        if model.config.prior_strength:
            if raw.shape[1]!=N_FEATURES:raise ValueError('Physics evidence requires the complete feature schema')
            pb,pc=physics_logits(raw);priors={'behaviour':pb,'composition':pc}
        z=model.forward(raw[:,:model.config.input_dim],priors)
        b.append(z['behaviour']);c.append(z['composition'])
    return np.vstack(b),np.vstack(c)


def probabilities(logits,temperature=1.):
    if not np.isfinite(temperature) or temperature<=0:raise ValueError('Temperature must be positive and finite')
    return softmax(np.asarray(logits)/temperature)


def temperatures(logits,targets,iterations=64):
    """Bounded golden-section search for held-out calibration NLL, no optimiser library."""
    z=np.asarray(logits,dtype=float);y=np.asarray(targets)
    if z.ndim!=2 or not len(z) or y.shape!=(len(z),) or not np.issubdtype(y.dtype,np.integer):raise ValueError('Invalid calibration arrays')
    if not np.isfinite(z).all() or ((y<0)|(y>=z.shape[1])).any():raise ValueError('Invalid calibration values')
    def objective(logt):
        scaled=z/np.exp(logt);shifted=scaled-scaled.max(1,keepdims=True)
        return float(np.mean(np.log(np.exp(shifted).sum(1))-shifted[np.arange(len(y)),y]))
    grid=np.linspace(-2.3,2.3,33);best=int(np.argmin([objective(t) for t in grid]))
    left=grid[max(best-1,0)];right=grid[min(best+1,len(grid)-1)];ratio=(np.sqrt(5)-1)/2
    a=right-ratio*(right-left);b=left+ratio*(right-left);fa=objective(a);fb=objective(b)
    for _ in range(iterations):
        if fa<fb:right,b,fb=b,a,fa;a=right-ratio*(right-left);fa=objective(a)
        else:left,a,fa=a,b,fb;b=left+ratio*(right-left);fb=objective(b)
    candidate=(left+right)/2
    return float(np.exp(candidate)) if objective(candidate)<=objective(0.) else 1.
