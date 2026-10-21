"""Matched validation search for learned physics reliability and robust weights."""
from dataclasses import asdict,fields
from pathlib import Path
import hashlib,time
import numpy as np
from .data import load,save_json
from .augmentation import append_development
from .features import N_BASE,N_FEATURES,FEATURE_NAMES
from .train import build_data,calibrate
from .scratch import NetworkConfig,TrainingConfig,TwoHeadMLP,fit as fit_network
from .model import save_model,load_model,predict_logits,probabilities
from .metrics import score,probability_metrics
from .quality import calibrate_guard

# Only measured quality/excitation indicators condition the gates.
GATE_FEATURES=(13,27,41,55,69,N_BASE+2,N_BASE+3,N_BASE+7,N_BASE+8,N_BASE+17,N_BASE+18,N_FEATURES-1)


def _train(d,run,training,seed,epochs=None):
    run=dict(run,seed=seed);options=training.copy()
    if epochs is not None:options['epochs']=epochs
    options['hidden_widths']=tuple(options['hidden_widths'])
    keys={f.name for f in fields(NetworkConfig)}-{'input_dim','seed','prior_strength','learned_physics_gate','gate_feature_indices'}
    network_options={k:options.pop(k) for k in list(options) if k in keys}
    net=NetworkConfig(N_FEATURES if run['enhanced'] else N_BASE,seed=seed,
                      prior_strength=run['prior_strength'],learned_physics_gate=run.get('learned_gate',False),
                      gate_feature_indices=GATE_FEATURES if run.get('learned_gate',False) else (),**network_options)
    config=TrainingConfig(net,physics_weight=run['physics_loss'],**options)
    train=build_data(d,np.flatnonzero(d['split']=='train'),run,True)
    mask=(d['split']=='validation')&(d['mask']==31)
    valid=build_data(d,np.flatnonzero(mask),run)
    model,history=fit_network(train,valid,config)
    b,c=predict_logits(model,d['X'][mask]);pb,pc=probabilities(b),probabilities(c)
    metrics=score(d['yb'][mask],d['yc'][mask],pb,pc)
    by_condition={condition:score(d['yb'][mask][d['condition'][mask]==condition],
                                 d['yc'][mask][d['condition'][mask]==condition],
                                 pb[d['condition'][mask]==condition],pc[d['condition'][mask]==condition])
                  for condition in np.unique(d['condition'][mask])}
    balanced=float(np.mean([r['joint_accuracy'] for r in by_condition.values()]))
    nll=probability_metrics(d['yc'][mask],pc)['nll']
    return model,history,{'validation':metrics,'validation_by_condition':by_condition,
        'balanced_validation_joint':balanced,'composition_nll':nll,'training_rows':len(train.x),
        'parameters':sum(p.size for p in model.params.values()),'config':asdict(config),
        'best_epoch':history['best_epoch'],'implementation':'manual_numpy'}


def _rank(entry):
    return (entry['balanced_validation_joint'],entry['validation']['composition_macro_f1'],-entry['composition_nll'])


def train_research(root,cfg,config_path,epochs=None):
    root=Path(root);art=root/'artifacts';art.mkdir(parents=True,exist_ok=True);(art/'search').mkdir(exist_ok=True)
    d=append_development(load(root),root,cfg['augmentation']['replicas'],cfg['augmentation']['seed'])
    search=[];summary=[];histories={}
    save_json(root/'reports/pretest_protocol.json',{'test_opened':False,'config':cfg,
        'selection':'mean joint accuracy over validation conditions; ties use composition macro-F1 then NLL',
        'search_budget_per_family':len(cfg['candidates']),'gate_feature_names':[FEATURE_NAMES[i] for i in GATE_FEATURES],
        'training_data':'same base and randomized training waveforms for every family'})
    for run in cfg['variants']:
        family=run['variant'];family_entries=[]
        for ci,candidate in enumerate(cfg['candidates']):
            started=time.perf_counter();training=dict(cfg['training'],**candidate)
            model,history,result=_train(d,run,training,11,epochs)
            name=f'{family}_candidate{ci}_11'
            result.update(name=name,variant=family,seed=11,candidate=ci,training_seconds=time.perf_counter()-started)
            save_model(art/f'search/{name}.npz',model,11,family)
            histories[name]=history;search.append(result);family_entries.append(result)
            print(f"{name}: balanced validation joint={result['balanced_validation_joint']:.4f}; {result['training_seconds']:.1f}s",flush=True)
        chosen=max(family_entries,key=_rank)
        for seed in cfg['training_seeds']:
            name=f'{family}_{seed}'
            if seed==11:
                result=chosen.copy();model,_=load_model(art/f"search/{chosen['name']}.npz");history=histories[chosen['name']]
            else:
                started=time.perf_counter()
                model,history,result=_train(d,run,dict(cfg['training'],**cfg['candidates'][chosen['candidate']]),seed,epochs)
                result.update(candidate=chosen['candidate'],training_seconds=time.perf_counter()-started)
            result.update(name=name,variant=family,seed=seed)
            save_model(art/f'{name}.npz',model,seed,family);histories[name]=history;summary.append(result)
            print(f"final {name}: validation={result['balanced_validation_joint']:.4f}",flush=True)
    selected=max(summary,key=_rank);model,meta=load_model(art/f"{selected['name']}.npz")
    save_model(art/'selected_model.npz',model,meta['seed'],meta['variant'])
    initial=TwoHeadMLP(model.config);initial.mean=model.mean.copy();initial.scale=model.scale.copy()
    save_model(art/'selected_initial_model.npz',initial,meta['seed'],'initial_parameters')
    calibration=calibrate(model,d);save_json(art/'calibration.json',calibration);calibrate_guard(root)
    save_json(root/'reports/search_results.json',{'candidates':search,'test_opened':False})
    save_json(root/'reports/training_history.json',histories)
    save_json(root/'reports/validation_results.json',{'runs':summary,'selected_neural':selected['name'],
            'selection_split':'validation only; equal candidate budget per family','test_opened':False,
            'implementation':'manual_numpy','calibration':calibration})
    save_json(root/'reports/parameter_learning.json',{'selected_neural':selected['name'],
            'learned_physics_gate':model.config.learned_physics_gate,
            'parameter_learning':histories[selected['name']]['parameter_learning'],
            'gate_feature_names':[FEATURE_NAMES[i] for i in model.config.gate_feature_indices],
            'optimizer_steps':histories[selected['name']]['optimizer_steps'],
            'best_epoch':histories[selected['name']]['best_epoch']})
    save_json(art/'artifact_contract.json',{'feature_names':FEATURE_NAMES,'implementation':'manual_numpy',
            'experiment_config':cfg,'config_sha256':hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
            'dataset_sha256':hashlib.sha256((root/'data/benchmark.npz').read_bytes()).hexdigest(),
            'augmentation_sha256':{s:hashlib.sha256((root/f'data/{s}_augmentation.npz').read_bytes()).hexdigest() for s in ['train','validation']},
            'selected_neural':selected['name'],'scope':'ideal controlled-source series-branch emulator; no hardware validation',
            'behaviours':['CC','CP','CZ'],'compositions':['R','L','C','RL','RC','LC','RLC']})
    print('Frozen selected model:',selected['name'],flush=True)
    return summary
