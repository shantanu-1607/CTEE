"""Waveform domain randomization restricted to declared development splits."""
from pathlib import Path
import json
import numpy as np
from .simulator import Circuit,BEHAVIOURS,supported,episode,transient,clean_probe
from .features import extract,N_FEATURES
from .data import save_json


def randomize(waves,startup,seed):
    """One sensor configuration and noise condition shared across an episode.

    Different channel errors are measurement errors, not changed physical labels.
    Fractional timing offset uses explicit periodic linear interpolation.
    """
    rng=np.random.default_rng(seed)
    gains=rng.uniform(.97,1.03,2);delay=float(rng.uniform(-1.,1.))
    snr=float(rng.uniform(12.,35.));alpha=float(rng.uniform(0.,.85))
    def transform(data):
        out=np.asarray(data,dtype=float).copy()*gains[:,None]
        low=int(np.floor(delay));fraction=delay-low
        out[...,1,:]=(1-fraction)*np.roll(out[...,1,:],low,axis=-1)+fraction*np.roll(out[...,1,:],low+1,axis=-1)
        noise=rng.normal(size=out.shape)
        for k in range(1,out.shape[-1]):noise[...,k]+=alpha*noise[...,k-1]
        noise/=np.maximum(np.sqrt(np.mean(noise**2,axis=-1,keepdims=True)),1e-12)
        noise*=np.sqrt(np.mean(out**2,axis=-1,keepdims=True))*10**(-snr/20)
        return (out+noise).astype(np.float32)
    return transform(waves),transform(startup),{'seed':int(seed),'gains':gains.tolist(),
                      'current_delay_samples':delay,'snr_db':snr,'noise_ar1':alpha}


def development_augmentation(manifest,split,replicas=3,seed=71):
    if split not in {'train','validation'}:raise ValueError('Domain randomization is restricted to development splits')
    if not isinstance(replicas,int) or replicas<1:raise ValueError('Use a positive augmentation count')
    rng=np.random.default_rng(seed);rows=[];ledger=[]
    for group,entry in enumerate(manifest['groups']):
        if entry['split']!=split:continue
        circuit=Circuit(**{k:entry[k] for k in Circuit.__dataclass_fields__})
        _,sv,si=transient(circuit,sample_rate=5000);base_start=np.stack([sv,si])*np.sqrt(2)
        for bi,behaviour in enumerate(BEHAVIOURS):
            if not supported(circuit,behaviour):continue
            w,f=episode(circuit,behaviour);start=base_start.copy();start[1]*=clean_probe(circuit,behaviour,0)[1]['gain']
            for replica in range(replicas):
                wave_seed=int(rng.integers(2**31));waves,ss,metadata=randomize(w,start,wave_seed)
                ledger.append(dict(group=group,behaviour=bi,replica=replica,**metadata))
                order=[0,1,2,3,4] if replica%2==0 else [0,2,3,1,4]
                for budget in range(1,6):
                    chosen=order[:budget]
                    rows.append((extract(waves,f,chosen,startup=ss),bi,group,sum(1<<p for p in chosen),
                                 ['R','L','C','RL','RC','LC','RLC'].index(circuit.composition)))
    if not rows:raise ValueError('No physical development groups found')
    return {'X':np.array([r[0] for r in rows]),'yb':np.array([r[1] for r in rows]),
            'group':np.array([r[2] for r in rows]),'mask':np.array([r[3] for r in rows]),
            'yc':np.array([r[4] for r in rows]),'split':np.full(len(rows),split),
            'condition':np.full(len(rows),'domain_randomized')},ledger


def append_development(d,root,replicas=3,seed=71):
    root=Path(root);manifest=json.loads((root/'data/manifest.json').read_text());pieces=[d];audit={}
    for offset,split in enumerate(['train','validation']):
        piece,ledger=development_augmentation(manifest,split,replicas,seed+offset)
        assert set(piece['group'])<=set(d['group'][d['split']==split])
        np.savez_compressed(root/f'data/{split}_augmentation.npz',**piece)
        audit[split]={'seed':seed+offset,'replicas':replicas,'records':len(piece['X']),
                      'physical_groups':len(set(piece['group'])),'episodes':ledger}
        pieces.append(piece)
    save_json(root/'reports/development_augmentation.json',audit)
    return {key:np.concatenate([part[key] for part in pieces]) for key in d}
