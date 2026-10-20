import numpy as np
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,roc_auc_score,log_loss

def probability_metrics(y,p,reference=None):
    k=p.shape[1];truth=np.eye(k)[y];confidence=p.max(1);correct=(p.argmax(1)==y)
    ece=0.
    for a,b in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
        mask=(confidence>a)&(confidence<=b)
        if mask.any():ece+=mask.mean()*abs(correct[mask].mean()-confidence[mask].mean())
    bs=float(np.mean(np.sum((p-truth)**2,axis=1)))
    result={'nll':float(log_loss(y,p,labels=np.arange(k))),'brier':bs,'ece_10_bins':float(ece)}
    if reference is not None:
        ref=np.repeat(np.asarray(reference)[None,:],len(y),axis=0)
        ref_bs=np.mean(np.sum((ref-truth)**2,axis=1))
        result['brier_skill']=float(1-bs/ref_bs) if ref_bs>0 else None
    try:result['macro_ovr_auc']=float(roc_auc_score(y,p,multi_class='ovr',average='macro',labels=np.arange(k)))
    except ValueError:result['macro_ovr_auc']=None
    return result

def score(yb,yc,pb,pc):
    b=pb.argmax(1);c=pc.argmax(1)
    return {'n':len(b),'behaviour_accuracy':float(accuracy_score(yb,b)),
            'composition_accuracy':float(accuracy_score(yc,c)),
            'behaviour_macro_f1':float(f1_score(yb,b,labels=range(3),average='macro',zero_division=0)),
            'composition_macro_f1':float(f1_score(yc,c,labels=range(7),average='macro',zero_division=0)),
            'joint_accuracy':float(np.mean((yb==b)&(yc==c))),
            'behaviour_confusion':confusion_matrix(yb,b,labels=range(3)).tolist(),
            'composition_confusion':confusion_matrix(yc,c,labels=range(7)).tolist()}

def group_bootstrap_difference(group,correct_a,correct_b,seed=19,repeats=1500):
    ids=np.unique(group);deltas=np.array([np.mean(correct_a[group==g]-correct_b[group==g]) for g in ids])
    rng=np.random.default_rng(seed)
    samples=np.mean(deltas[rng.integers(0,len(ids),(repeats,len(ids)))],axis=1)
    # A paired random-sign permutation uses physical configurations as units.
    null=np.mean(deltas[None,:]*rng.choice([-1,1],size=(5000,len(ids))),axis=1)
    return {'difference':float(deltas.mean()),'ci95':np.quantile(samples,[.025,.975]).tolist(),
            'paired_group_permutation_p':float((1+np.sum(abs(null)>=abs(deltas.mean())))/(1+len(null))),
            'independent_groups':len(ids),'unit':'parent physical branch, averaging all its supported modes'}

