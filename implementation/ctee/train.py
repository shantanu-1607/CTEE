"""Manual NumPy training on physical-group-separated electrical measurements."""
from dataclasses import asdict
from pathlib import Path
import hashlib,json,time
import numpy as np
from .data import load,save_json
from .features import N_BASE,N_FEATURES,FEATURE_NAMES,physics_logits
from .model import save_model,load_model,predict_logits,probabilities,temperatures
from .metrics import score,probability_metrics
from .scratch import NetworkConfig,TrainingConfig,TrainingData,fit as fit_network
from .runtime import settings,DEFAULT_CONFIG


def build_data(d,indices,run,training=False):
    raw=d['X'][indices].astype(float,copy=True)
    if training and run['enhanced']:
        absent=np.random.default_rng(run['seed']+1000).random(len(raw))<.30
        raw[absent,-6:]=0
    priors=None
    if run['prior_strength']:
        b,c=physics_logits(raw);priors={'behaviour':b,'composition':c}
    return TrainingData(raw if run['enhanced'] else raw[:,:N_BASE],d['yb'][indices],d['yc'][indices],
                        behaviour_mask=raw[:,N_BASE+3]>.5,slopes=raw[:,N_BASE],
                        slope_mask=raw[:,N_BASE+3]>.5,priors=priors)


def calibrate(model,d):
    all_cal=d['split']=='calibration';budgets=np.array([int(m).bit_count() for m in d['mask']])
    result={'implementation':'manual_numpy','per_budget':{},'target_accuracy':.95,
            'method':'Temperatures and thresholds fitted only on calibration configurations, separately by probe count. This is not a universal OOD guarantee.'}
    for budget in range(1,6):
        mask=all_cal&(budgets==budget);x=d['X'][mask];yb=d['yb'][mask];yc=d['yc'][mask]
        b,c=predict_logits(model,x);tb=temperatures(b,yb);tc=temperatures(c,yc)
        pb=probabilities(b,tb);pc=probabilities(c,tc);correct=(pb.argmax(1)==yb)&(pc.argmax(1)==yc)
        gate=(x[:,N_BASE+3]>.5)&(x[:,N_BASE+8]>.5)&(x[:,N_BASE+7]<.2)
        conf=np.minimum(pb.max(1),pc.max(1));threshold=1.01;coverage=0.;accuracy=None
        for t in np.linspace(.5,.995,100):
            accepted=gate&(conf>=t)
            if accepted.sum()>=20 and correct[accepted].mean()>=.95 and accepted.mean()>coverage:
                threshold=float(t);coverage=float(accepted.mean());accuracy=float(correct[accepted].mean())
        result['per_budget'][str(budget)]={'behaviour_temperature':tb,'composition_temperature':tc,'threshold':threshold,
            'calibration_coverage':coverage,'calibration_accepted_joint_accuracy':accuracy,
            'before':{'behaviour':probability_metrics(yb,probabilities(b)),'composition':probability_metrics(yc,probabilities(c))},
            'after':{'behaviour':probability_metrics(yb,pb),'composition':probability_metrics(yc,pc)}}
    result.update(result['per_budget']['5'])
    return result


def fit(root,epochs=None,config_path=None):
    root=Path(root);out=root/'artifacts';out.mkdir(parents=True,exist_ok=True)
    cfg=settings(config_path);d=load(root)
    if d['X'].shape[1]!=N_FEATURES:raise ValueError('Regenerate the NumPy benchmark; legacy features have a different schema')
    validation=(d['split']=='validation')&(d['mask']==31);valx=d['X'][validation]
    summary=[];histories={}
    for run in cfg['runs']:
        started=time.perf_counter();options=cfg['training'].copy();widths=tuple(options.pop('hidden_widths'))
        network_args={name:options.pop(name) for name in ['activation','negative_slope','dropout','initialization','dtype']}
        if epochs is not None:options['epochs']=epochs
        config=TrainingConfig(NetworkConfig(N_FEATURES if run['enhanced'] else N_BASE,widths,
                         seed=run['seed'],prior_strength=run['prior_strength'],**network_args),physics_weight=run['physics_loss'],**options)
        use=(d['split']=='train')&np.isin(d['condition'],['clean','noise30','noise20'] if run['robust'] else ['clean'])
        indices=np.flatnonzero(use)
        if not run['robust']:indices=np.tile(indices,3) # equal update budget, not additional independent devices
        train=build_data(d,indices,run,True);valid=build_data(d,np.flatnonzero(validation),run)
        model,history=fit_network(train,valid,config)
        b,c=predict_logits(model,valx);metrics=score(d['yb'][validation],d['yc'][validation],probabilities(b),probabilities(c))
        name=f"{run['variant']}_{run['seed']}";save_model(out/f'{name}.npz',model,run['seed'],run['variant'])
        entry={'name':name,'variant':run['variant'],'seed':run['seed'],'validation':metrics,
               'training_rows':len(indices),'best_epoch':history['best_epoch'],'training_seconds':time.perf_counter()-started,
               'parameters':sum(p.size for p in model.params.values()),'config':asdict(config),'implementation':'manual_numpy'}
        summary.append(entry);histories[name]=history
        print(f"{name}: validation joint={metrics['joint_accuracy']:.3f}; best epoch={history['best_epoch']}; {entry['training_seconds']:.1f}s",flush=True)
    selected=max(summary,key=lambda r:(r['validation']['joint_accuracy'],r['validation']['composition_macro_f1']))
    model,meta=load_model(out/f"{selected['name']}.npz")
    save_model(out/'selected_model.npz',model,meta['seed'],meta['variant'])
    calibration=calibrate(model,d);save_json(out/'calibration.json',calibration)
    save_json(root/'reports/training_history.json',histories)
    save_json(root/'reports/validation_results.json',{'runs':summary,'selected_neural':selected['name'],
              'selection_split':'validation only','test_opened':False,'implementation':'manual_numpy','calibration':calibration})
    save_json(out/'artifact_contract.json',{'feature_names':FEATURE_NAMES,'behaviours':['CC','CP','CZ'],
              'compositions':['R','L','C','RL','RC','LC','RLC'],'implementation':'manual_numpy',
              'dataset_sha256':hashlib.sha256((root/'data/benchmark.npz').read_bytes()).hexdigest(),
              'experiment_config':cfg,'config_sha256':hashlib.sha256(Path(config_path or DEFAULT_CONFIG).read_bytes()).hexdigest(),
              'selection_split':'validation','calibration_split':'calibration','selected_neural':selected['name'],
              'scope':'ideal controlled-source simulator; known passive branch; no hardware validation'})
    from .quality import calibrate_guard
    calibrate_guard(root)
    return summary
