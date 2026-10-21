from pathlib import Path
import time,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import load,save_json,CONDITIONS
from .model import load_model,predict_logits,probabilities
from .features import physics_logits,extract,N_BASE
from .metrics import score,probability_metrics,group_bootstrap_difference,continuous_metrics
from .statistics import compare_panels
from .simulator import *
from .inference import Predictor,trace_policy
from .gate_analysis import analyze_gates

def panels_analysis(group,labels,correct):
    ids=np.unique(group);rng=np.random.default_rng(94)
    count=min(10,min(sum(labels[group==g][0]==c for g in ids) for c in np.unique(labels)))
    if count<2:return {'limitation':'At least two independent panels per class are required'}
    panels=[[] for _ in range(count)]
    for composition in np.unique(labels):
        cls=[g for g in ids if labels[group==g][0]==composition]
        rng.shuffle(cls)
        for i,g in enumerate(cls):panels[i%count].append(g)
    names=list(correct)
    matrix=np.array([[np.mean(correct[n][np.isin(group,panel)]) for n in names] for panel in panels])
    result=compare_panels(matrix,names)
    result.update(groups_per_panel=[len(p) for p in panels],panel_accuracies=matrix.tolist(),
                  limitation='Stratified disjoint held-out simulator device panels, conditional on fixed trained models. Not cross-validation folds, independent laboratories, or training seeds.')
    return result

def evaluate(root):
    root=Path(root);out=root/'reports';fig=out/'figures';fig.mkdir(parents=True,exist_ok=True)
    d=load(root);validation=json.loads((out/'validation_results.json').read_text())
    test=(d['split']=='test')&(d['mask']==31)
    x=d['X'][test];yb=d['yb'][test];yc=d['yc'][test];cond=d['condition'][test];groups=d['group'][test]
    results={};predictions={};gate_results={}
    for entry in validation['runs']:
        name=entry['name']
        model,_=load_model(root/f'artifacts/{name}.npz');b,c=predict_logits(model,x);pb=probabilities(b);pc=probabilities(c)
        if model.config.learned_physics_gate:
            gate_results[name]=analyze_gates(model,x,yb,yc,cond,groups)
        predictions[name]=(pb,pc)
        results[name]={s:score(yb[cond==s],yc[cond==s],pb[cond==s],pc[cond==s]) for s in np.unique(cond)}
    rb,rc=physics_logits(x);predictions['electrical_rules']=(probabilities(rb),probabilities(rc))
    results['electrical_rules']={s:score(yb[cond==s],yc[cond==s],probabilities(rb[cond==s]),probabilities(rc[cond==s])) for s in np.unique(cond)}
    selected=validation['selected_neural'];pb,pc=predictions[selected]
    baseb,basec=predictions['baseline_robust_11']
    correct=((pb.argmax(1)==yb)&(pc.argmax(1)==yc)).astype(float)
    base_correct=((baseb.argmax(1)==yb)&(basec.argmax(1)==yc)).astype(float)
    mask=cond=='noise20'
    comparison=group_bootstrap_difference(groups[mask],correct[mask],base_correct[mask])
    names=['baseline_robust_11','features_only_11','physics_enhanced_11','physics_gated_11',selected,'electrical_rules']
    names=[n for n in dict.fromkeys(names) if n in predictions]
    panel_correct={n:((predictions[n][0].argmax(1)==yb)&(predictions[n][1].argmax(1)==yc)).astype(float)[mask] for n in names}
    statistics=panels_analysis(groups[mask],yc[mask],panel_correct)
    predictor=Predictor(root);cal=predictor.calibration
    chosen_model=predictor.model;lb,lc=predict_logits(chosen_model,x)
    calibrated_b=probabilities(lb,cal['behaviour_temperature']);calibrated_c=probabilities(lc,cal['composition_temperature'])
    trainfull=(d['split']=='train')&(d['mask']==31)
    refb=np.bincount(d['yb'][trainfull],minlength=3)/sum(trainfull);refc=np.bincount(d['yc'][trainfull],minlength=7)/sum(trainfull)
    probability_results={s:{'before':{'behaviour':probability_metrics(yb[cond==s],pb[cond==s],refb),'composition':probability_metrics(yc[cond==s],pc[cond==s],refc)},
                           'after':{'behaviour':probability_metrics(yb[cond==s],calibrated_b[cond==s],refb),'composition':probability_metrics(yc[cond==s],calibrated_c[cond==s],refc)}} for s in np.unique(cond)}
    regression={}
    for s in np.unique(cond):
        m=cond==s;truth=np.array([0.,-1.,1.])[yb[m]];estimate=x[m,N_BASE]
        regression[s]=continuous_metrics(truth,estimate)
    # Raw predictions and all truths are exported for independent re-analysis.
    np.savez_compressed(out/'heldout_predictions.npz',yb=yb,yc=yc,groups=groups,conditions=cond,
                        pb=calibrated_b,pc=calibrated_c,baseline_pb=baseb,baseline_pc=basec)
    seed_summary={}
    for family in dict.fromkeys(r['variant'] for r in validation['runs']):
        entries=[r for r in validation['runs'] if r['variant']==family]
        seed_summary[family]={condition:{'mean':float(np.mean([results[r['name']][condition]['joint_accuracy'] for r in entries])),
                                        'sample_sd':float(np.std([results[r['name']][condition]['joint_accuracy'] for r in entries],ddof=1)) if len(entries)>1 else None,
                                        'seeds':[r['seed'] for r in entries]}
                              for condition in np.unique(cond)}
    if gate_results:save_json(out/'gate_counterfactual.json',gate_results)
    save_json(out/'test_results.json',{'selected_neural':selected,'conditions':results,'comparison_at_20dB':comparison,
             'implementation':'manual_numpy','training_seed_summary':seed_summary,'probability_metrics':probability_results,'continuous_estimation':regression,'statistics':statistics,
             'scope':'held-out configurations from the same ideal controlled-source generator; not appliance or hardware validation'})
    plot_results(results,selected,fig)
    plot_confusions(results[selected]['noise20'],fig)
    confidence=np.minimum(calibrated_b[mask].max(1),calibrated_c[mask].max(1))
    curve=[]
    for t in np.linspace(.4,.999,100):
        accept=confidence>=t
        curve.append({'threshold':float(t),'coverage':float(accept.mean()),'risk':float(1-correct[mask][accept].mean()) if accept.any() else None})
    save_json(out/'risk_coverage.json',curve)
    policy_results=evaluate_policies(root,predictor)
    latency=benchmark_latency(root,predictor)
    write_summary(root,results,selected,comparison,policy_results,latency)
    print('Test results:',json.dumps({n:{s:round(r['joint_accuracy'],4) for s,r in values.items()} for n,values in results.items()},indent=2),flush=True)
    return results

def plot_results(results,selected,fig):
    names=['baseline_clean_11','baseline_robust_11','features_only_11','physics_enhanced_11','physics_gated_11',selected,'electrical_rules'];names=[n for n in dict.fromkeys(names) if n in results]
    conditions=['clean','noise20','sensor_shift','frequency_shift','coloured10']
    ax=plt.figure(figsize=(10,5)).subplots()
    for n in names:ax.plot(conditions,[results[n][s]['joint_accuracy'] for s in conditions],marker='o',label=n)
    ax.set(ylabel='Both labels correct (fraction)',ylim=(0,1.02),title='Held-out synthetic physical configurations')
    ax.legend(fontsize=8);ax.grid(alpha=.2);plt.tight_layout();plt.savefig(fig/'robustness.png',dpi=180);plt.close()

def plot_confusions(r,fig):
    figobj,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,key,labels in zip(axes,['behaviour_confusion','composition_confusion'],[BEHAVIOURS,COMPOSITIONS]):
        arr=np.array(r[key]);ax.imshow(arr,cmap='Blues');ax.set_xticks(range(len(labels)),labels);ax.set_yticks(range(len(labels)),labels)
        ax.set(xlabel='Predicted',ylabel='True',title='20 dB held-out test')
        for i in range(len(labels)):
            for j in range(len(labels)):ax.text(j,i,str(arr[i,j]),ha='center',va='center',fontsize=8)
    plt.tight_layout();plt.savefig(fig/'confusion_matrices.png',dpi=180);plt.close()

def evaluate_policies(root,predictor):
    manifest=json.loads((root/'data/manifest.json').read_text())
    records=[]
    for gi,entry in enumerate(manifest['groups']):
        if entry['split']!='test':continue
        circuit=Circuit(**{k:entry[k] for k in Circuit.__dataclass_fields__})
        _,sv,si=transient(circuit,sample_rate=5000);startup=np.stack([sv,si])*np.sqrt(2)
        for bi,b in enumerate(BEHAVIOURS):
            if not supported(circuit,b):continue
            gain=clean_probe(circuit,b,0)[1]['gain'];start=startup.copy();start[1]*=gain
            for condition in ['clean','noise20','coloured10']:
                snr,kind=CONDITIONS[condition];seed=9823+gi*100+bi
                waves,fs=episode(circuit,b,seed,snr,kind);ss=corrupt(start,np.random.default_rng(seed),snr,kind)
                for policy in ['fixed','random','adaptive']:
                    result=trace_policy(predictor,waves,fs,ss,policy,seed,diagnostic=True)
                    decision=result['decision']
                    records.append({'group':gi,'condition':condition,'policy':policy,
                       'accepted':decision['accepted'],'probes':decision['budget'],
                       'correct':decision['behaviour']==bi and decision['composition']==COMPOSITIONS.index(circuit.composition),
                       'curve_correct':[r['behaviour']==bi and r['composition']==COMPOSITIONS.index(circuit.composition) for r in result['trace']]})
    summaries={}
    for condition in ['clean','noise20','coloured10']:
        summaries[condition]={}
        for policy in ['fixed','random','adaptive']:
            rr=[r for r in records if r['condition']==condition and r['policy']==policy]
            accept=[r for r in rr if r['accepted']]
            summaries[condition][policy]={'episodes':len(rr),'coverage':len(accept)/len(rr),
               'accepted_joint_accuracy':float(np.mean([r['correct'] for r in accept])) if accept else None,
               'unconditional_decision_joint_accuracy':float(np.mean([r['correct'] for r in rr])),
               'mean_probes':float(np.mean([r['probes'] for r in rr])),
               'mean_recording_seconds_lower_bound':float(.3+DURATION*np.mean([r['probes'] for r in rr])),
               'joint_accuracy_by_probe_budget':np.mean([r['curve_correct'] for r in rr],axis=0).tolist()}
    save_json(root/'reports/acquisition_results.json',{'results':summaries,'records':records,
        'limitation':'Fixed/random/adaptive use identical per-episode waveforms and frozen model. Temperature and threshold calibrated separately for each probe count; reliability outside the calibration distribution is not guaranteed. Diagnostic traces continue after the operational stop only to draw matched-budget curves. Recording time omits settling/actuation.'})
    fig,ax=plt.subplots(figsize=(7,4))
    for policy in ['fixed','random','adaptive']:
        ax.plot(range(1,6),summaries['noise20'][policy]['joint_accuracy_by_probe_budget'],marker='o',label=policy)
    ax.set(xlabel='Number of observations',ylabel='Both labels correct',title='20 dB: matched observation budgets',ylim=(0,1.02));ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(root/'reports/figures/acquisition.png',dpi=180);plt.close(fig)
    return summaries

def benchmark_latency(root,predictor):
    c=sample_circuit('RLC',0,np.random.default_rng(20));w,f=episode(c,'CP',23,20)
    times=[];infer=[]
    for _ in range(100):
        t=time.perf_counter();predictor.predict(w,f,list(range(5)));times.append((time.perf_counter()-t)*1000)
        x=extract(w,f);mid=time.perf_counter();predictor.predict_features(x);infer.append((time.perf_counter()-mid)*1000)
    result={'feature_plus_model_ms':{'p50':float(np.percentile(times,50)),'p95':float(np.percentile(times,95))},
            'model_ms':{'p50':float(np.percentile(infer,50)),'p95':float(np.percentile(infer,95))},
            'n':100,'device':'local CPU','includes_acquisition':False,'includes_UI':False,
            'model_bytes':(root/'artifacts/selected_model.npz').stat().st_size}
    save_json(root/'reports/latency.json',result);return result

def write_summary(root,results,selected,comparison,policy,latency):
    lines=['# Measured results - research prototype','',f'Selected integrated model: **{selected}**, chosen on validation data before final testing.',
           '', '**Scope:** synthetic controlled-source emulator; no claim of appliance/hardware accuracy or global best classifier.','',
           '| Model | Clean joint accuracy | 20 dB joint accuracy | Sensor shift | Frequency shift | Coloured 10 dB |',
           '|---|---:|---:|---:|---:|---:|']
    for name,rr in results.items():lines.append('| '+name+' | '+' | '.join(f"{rr[s]['joint_accuracy']:.2%}" for s in ['clean','noise20','sensor_shift','frequency_shift','coloured10'])+' |')
    lines+=['',f"20 dB gain over noise-trained baseline (physical-device average): {comparison['difference']:.2%}; physical-group bootstrap 95% CI {comparison['ci95']}. Episode-pooled gain: {comparison['pooled_joint_difference']:.2%}.",
            '', '## Acquisition at 20 dB','', '| Policy | Mean probes | Coverage | Accepted joint accuracy |','|---|---:|---:|---:|']
    for p,r in policy['noise20'].items():lines.append(f"| {p} | {r['mean_probes']:.2f} | {r['coverage']:.2%} | {r['accepted_joint_accuracy']:.2%} |" if r['accepted_joint_accuracy'] is not None else f"| {p} | {r['mean_probes']:.2f} | 0% | undefined |")
    lines+=['',f"Feature + model compute p95: {latency['feature_plus_model_ms']['p95']:.2f} ms on the local CPU, excluding acquisition and UI.",
            '', '## Interpretation','',
            'All compared neural families use explicit NumPy training. Use the noise-trained baseline, matched development augmentation and the same validation-search budget for comparisons. Inspect the features-only, fixed-physics and learned-gate ablations before crediting an individual component. The selected checkpoint is chosen on validation, not by the best test seed.',
            '', 'Seed means and standard deviations, physical-device bootstrap intervals and panel tests are exported in test_results.json. The gate_counterfactual.json file, when present, tests the learned gates against constant half-valued gates with identical frozen neural weights. Some differences may be inconclusive. Test accuracy and accepted-set accuracy are distinct quantities.',
            '', 'The input-quality guard and calibration-by-budget were fitted on calibration devices before this test. A fresh independent configuration pool checks the frozen classifier and guard in fresh_confirmation.json. Low coverage/rejection is not increased classification accuracy.',
            '', 'Independent ngspice verification, broader controller families, real datasets with verified dual labels, hardware, and external generalisation remain open validation work. See all JSON/NPZ exports for exact values and failure cases.']
    (root/'reports/RESULTS.md').write_text('\n'.join(lines)+'\n')
