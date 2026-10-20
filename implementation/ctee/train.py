from pathlib import Path
import hashlib
import time,json
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.ensemble import RandomForestClassifier
import joblib
from .data import load,save_json
from .features import N_BASE,FEATURE_NAMES
from .model import SharedClassifier,save_model,load_model,predict_logits,probabilities,temperatures
from .metrics import score,probability_metrics

RUNS=[('baseline_clean',False,0.,False,11),('baseline_robust',False,0.,True,11),
      ('features_only',True,0.,True,11),('physics_enhanced',True,.5,True,11),
      ('physics_enhanced',True,.5,True,22),('physics_enhanced',True,.5,True,33)]

def fit(root,epochs=100):
    root=Path(root);out=root/'artifacts';out.mkdir(exist_ok=True)
    d=load(root);torch.set_num_threads(2)
    validation=(d['split']=='validation')&(d['mask']==31)
    valx=d['X'][validation];ybv=d['yb'][validation];ycv=d['yc'][validation]
    summary=[];history={}
    for variant,enhanced,physics,robust,seed in RUNS:
        start=time.perf_counter();torch.manual_seed(seed);rng=np.random.default_rng(seed)
        use=(d['split']=='train')&(np.isin(d['condition'],['clean','noise30','noise20']) if robust else (d['condition']=='clean'))
        x=torch.tensor(d['X'][use]);yb=torch.tensor(d['yb'][use]);yc=torch.tensor(d['yc'][use])
        if enhanced:
            # CSV recordings may legitimately omit the optional startup capture.
            absent=rng.random(len(x))<.30
            x[absent,-6:]=0
        model=SharedClassifier(enhanced,physics)
        n=model.mean.numel();model.mean.copy_(x[:,:n].mean(0));model.scale.copy_(x[:,:n].std(0).clamp_min(.01))
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        best=-1.;patience=0;curve=[];name=f'{variant}_{seed}';path=out/f'{name}.pt'
        for epoch in range(epochs):
            model.train();total=0.
            batches=np.array_split(rng.permutation(len(x)),max(1,int(np.ceil(len(x)/512))))
            for ids in batches:
                xb=x[ids];b,c=model(xb)
                valid=xb[:,N_BASE+3]>.5
                loss=F.cross_entropy(c,yc[ids])
                if valid.any():loss=loss+F.cross_entropy(b[valid],yb[ids][valid])
                if physics and valid.any():
                    predicted=F.softmax(b[valid],dim=1)@torch.tensor([0.,-1.,1.])
                    loss=loss+.03*F.smooth_l1_loss(predicted,xb[valid,N_BASE])
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
                total+=float(loss.detach())*len(ids)
            model.eval();vb,vc=predict_logits(model,valx)
            result=score(ybv,ycv,probabilities(vb),probabilities(vc))
            value=result['joint_accuracy']+.05*(result['behaviour_macro_f1']+result['composition_macro_f1'])
            curve.append({'epoch':epoch+1,'loss':total/len(x),'validation_joint_accuracy':result['joint_accuracy']})
            if value>best+1e-5:
                best=value;best_result=result;best_epoch=epoch+1;save_model(path,model,seed,variant);patience=0
            else:patience+=1
            if patience>=16:break
        entry={'name':name,'variant':variant,'seed':seed,'enhanced_features':enhanced,'robust_training':robust,
               'physics_weight':physics,'best_epoch':best_epoch,'training_rows':len(x),
               'training_seconds':time.perf_counter()-start,'validation':best_result,'parameters':sum(p.numel() for p in model.parameters())}
        summary.append(entry);history[name]=curve
        print(f"{name}: val joint={best_result['joint_accuracy']:.3f}, epoch={best_epoch}, {entry['training_seconds']:.1f}s",flush=True)
    # These classical comparators receive the same full observed records, no labels as features.
    tr=(d['split']=='train')&(d['mask']==31)
    for label,cols in [('rf_baseline',N_BASE),('rf_enhanced',d['X'].shape[1])]:
        rf=RandomForestClassifier(n_estimators=240,min_samples_leaf=2,max_features=.8,n_jobs=2,random_state=11)
        rf.fit(d['X'][tr,:cols],np.column_stack([d['yb'][tr],d['yc'][tr]]))
        joblib.dump(rf,out/f'{label}.joblib')
        bp,cp=rf.predict_proba(valx[:,:cols]);r=score(ybv,ycv,bp,cp)
        summary.append({'name':label,'variant':'random_forest','seed':11,'validation':r,'feature_count':cols})
    # Select only on validation data. The main deployment remains one integrated neural model.
    neural=[r for r in summary if r['variant']!='random_forest']
    selected=max(neural,key=lambda r:(r['validation']['joint_accuracy'],r['validation']['composition_macro_f1']))
    model,saved=load_model(out/f"{selected['name']}.pt")
    save_model(out/'selected_model.pt',model,saved['seed'],saved['variant'])
    cal=(d['split']=='calibration')&(d['mask']==31)
    cb,cc=predict_logits(model,d['X'][cal]);yb=d['yb'][cal];yc=d['yc'][cal]
    tb=temperatures(cb,yb);tc=temperatures(cc,yc)
    pb=probabilities(cb,tb);pc=probabilities(cc,tc)
    correct=(pb.argmax(1)==yb)&(pc.argmax(1)==yc)
    gate=(d['X'][cal,N_BASE+3]>.5)&(d['X'][cal,N_BASE+8]>.5)&(d['X'][cal,N_BASE+7]<.2)
    conf=np.minimum(pb.max(1),pc.max(1));threshold=1.01;coverage=0.;accuracy=None
    for t in np.linspace(.5,.995,100):
        accepted=gate&(conf>=t)
        if accepted.sum()>=20 and correct[accepted].mean()>=.95 and accepted.mean()>coverage:
            threshold=float(t);coverage=float(accepted.mean());accuracy=float(correct[accepted].mean())
    calibration={'behaviour_temperature':tb,'composition_temperature':tc,'threshold':threshold,
                 'calibration_coverage':coverage,'calibration_accepted_joint_accuracy':accuracy,
                 'method':'full-library calibration; partial-probe calibration is not guaranteed',
                 'before':{'behaviour':probability_metrics(yb,probabilities(cb)),'composition':probability_metrics(yc,probabilities(cc))},
                 'after':{'behaviour':probability_metrics(yb,pb),'composition':probability_metrics(yc,pc)}}
    save_json(out/'calibration.json',calibration)
    save_json(root/'reports/training_history.json',history)
    save_json(root/'reports/validation_results.json',{'runs':summary,'selected_neural':selected['name'],
              'selection_split':'validation only','test_opened':False,'calibration':calibration})
    save_json(out/'artifact_contract.json',{'feature_names':FEATURE_NAMES,'behaviours':['CC','CP','CZ'],
              'compositions':['R','L','C','RL','RC','LC','RLC'],'dataset_sha256':hashlib.sha256((root/'data/benchmark.npz').read_bytes()).hexdigest(),
              'selection_split':'validation','calibration_split':'calibration','selected_neural':selected['name'],
              'runs':[list(r) for r in RUNS],'max_epochs':epochs,'optional_startup_training_dropout':.30,
              'scope':'docs/LABEL_CONTRACT.md'})
    from .quality import calibrate_guard
    calibrate_guard(root)
    return summary

