"""Group-safe benchmark generation, with raw examples and complete provenance."""
from pathlib import Path
import hashlib,json,time
import numpy as np
from .simulator import *
from .features import extract, FEATURE_NAMES

CONDITIONS={"clean":(None,"nominal"),"noise30":(30,"nominal"),"noise20":(20,"nominal"),
            "sensor_shift":(20,"sensor_shift"),"frequency_shift":(20,"frequency_shift"),"coloured10":(10,"coloured")}

def save_json(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False))

def split_indices(n,rng):
    if n<20:raise ValueError("Use at least 20 independent configurations per topology")
    perm=rng.permutation(n);a=int(.6*n);b=int(.75*n);c=int(.85*n)
    return {int(i):part for part,ids in [('train',perm[:a]),('validation',perm[a:b]),('calibration',perm[b:c]),('test',perm[c:])] for i in ids}

def generate(root,configurations=100,seed=20261020):
    root=Path(root);out=root/'data';out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed)
    rows=[];yb=[];yc=[];groups=[];parts=[];conditions=[];masks=[]
    manifest=[];example_ids=[];episode_seeds=[]
    started=time.perf_counter()
    for ci,composition in enumerate(COMPOSITIONS):
        splits=split_indices(configurations,rng)
        for index in range(configurations):
            circuit=sample_circuit(composition,index,rng)
            group=len(manifest)
            entry=circuit.to_dict();entry['split']=splits[index]
            physical={k:entry[k] for k in ['resistance','inductance','capacitance','parasitic_resistance']}
            entry['parameter_hash']=hashlib.sha256(json.dumps(physical,sort_keys=True).encode()).hexdigest()
            manifest.append(entry)
            _,sv,si=transient(circuit,duration=.3,sample_rate=5000)
            startup=np.stack([sv,si])*np.sqrt(2)
            for bi,behaviour in enumerate(BEHAVIOURS):
                if not supported(circuit,behaviour):continue
                gain=clean_probe(circuit,behaviour,0)[1]['gain']
                mode_startup=startup.copy();mode_startup[1]*=gain
                for condition,(snr,corruption) in CONDITIONS.items():
                    if entry['split'] in ['train','validation','calibration'] and condition not in ['clean','noise30','noise20']:continue
                    sim_seed=int(rng.integers(0,2**31))
                    waves,fs=episode(circuit,behaviour,sim_seed,snr,corruption)
                    episode_seeds.append({'group':group,'behaviour':bi,'condition':condition,'wave_seed':sim_seed,'startup_rng_state':rng.bit_generator.state.copy()})
                    noisy_start=corrupt(mode_startup,rng,snr,corruption)
                    # A variety of legal acquisition masks, all within this parent group.
                    order=[0,1,2,3,4]
                    if (index+bi)%2:order=[0,2,3,1,4]
                    selections=[order[:k] for k in range(1,6)]
                    for selected in selections:
                        rows.append(extract(waves,fs,selected,startup=noisy_start))
                        yb.append(bi);yc.append(ci);groups.append(group);parts.append(entry['split']);conditions.append(condition)
                        masks.append(sum(1<<p for p in selected))
                    if entry['split']=='test' and condition=='noise20' and sum(s.startswith(composition+'_') for s in example_ids)<6:
                        name=f'{composition}_{behaviour}_{index}'
                        np.savez_compressed(out/f'example_{name}.npz',waves=waves,frequencies=fs,startup=noisy_start,
                                            behaviour=bi,composition=ci,group=group)
                        example_ids.append(name)
            if (index+1)%25==0:print(f'  {composition}: {index+1}/{configurations}',flush=True)
    np.savez_compressed(out/'benchmark.npz',X=np.asarray(rows,dtype=np.float32),yb=np.asarray(yb),yc=np.asarray(yc),
                        group=np.asarray(groups),split=np.asarray(parts),condition=np.asarray(conditions),mask=np.asarray(masks))
    save_json(out/'manifest.json',{'seed':seed,'configurations_per_topology':configurations,'feature_names':FEATURE_NAMES,
              'groups':manifest,'episodes':episode_seeds,'implementation':'manual_numpy','examples':example_ids,'supported_pairs':SUPPORT,'conditions':CONDITIONS,
              'generation_seconds':time.perf_counter()-started,'label_contract':'docs/LABEL_CONTRACT.md'})
    print(f'Generated {len(manifest)} physical configurations, {len(rows)} feature records in {time.perf_counter()-started:.1f}s',flush=True)
    return audit_splits(root)

def load(root):
    with np.load(Path(root)/'data/benchmark.npz',allow_pickle=False) as d:return {k:d[k] for k in d.files}

def audit_splits(root):
    d=load(root);parts=np.unique(d['split']);sets={p:set(d['group'][d['split']==p].tolist()) for p in parts}
    for i,a in enumerate(parts):
        for b in parts[i+1:]:assert not sets[a]&sets[b],f'Group leakage: {a}/{b}'
    assert np.isfinite(d['X']).all()
    report={'status':'passed','physical_groups':len(set(d['group'].tolist())),
            'records':len(d['X']),'splits':{p:{'groups':len(sets[p]),'records':int(sum(d['split']==p))} for p in parts},
            'group_overlap':0,'feature_count':d['X'].shape[1],
            'test_used_for_training':False,'unsupported_pairs':int(sum((d['yb']==1)&~np.isin(d['yc'],[0,3,4,6])))}
    assert report['unsupported_pairs']==0
    save_json(Path(root)/'reports/data_audit.json',report)
    return report

