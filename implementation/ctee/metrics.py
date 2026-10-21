"""Classification, probability, and continuous metrics computed with NumPy."""
import numpy as np


def _inputs(y, p):
    y = np.asarray(y); p = np.asarray(p, dtype=float)
    if p.ndim != 2 or not len(p) or y.shape != (len(p),) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError('Expected nonempty probability rows and integer labels')
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1., atol=1e-5):
        raise ValueError('Probabilities must be finite, nonnegative, and sum to one')
    if ((y < 0) | (y >= p.shape[1])).any(): raise ValueError('Label outside class range')
    return y, p


def confusion(y, predicted, classes):
    matrix = np.zeros((classes, classes), dtype=np.int64)
    for actual, guess in zip(y, predicted): matrix[int(actual), int(guess)] += 1
    return matrix


def macro_f1(matrix):
    diagonal = np.diag(matrix).astype(float); denominator = matrix.sum(0) + matrix.sum(1)
    return float(np.mean(np.divide(2 * diagonal, denominator, out=np.zeros_like(diagonal), where=denominator > 0)))


def binary_auc(truth, scores):
    """Mann-Whitney AUC with explicitly averaged ranks for tied scores."""
    truth=np.asarray(truth,dtype=bool); scores=np.asarray(scores,dtype=float)
    if truth.shape!=scores.shape or truth.ndim!=1 or not np.isfinite(scores).all(): raise ValueError('Invalid AUC inputs')
    positives=int(truth.sum()); negatives=len(truth)-positives
    if not positives or not negatives: return None
    order=np.argsort(scores,kind='stable'); ranks=np.empty(len(scores),dtype=float); start=0
    while start<len(scores):
        end=start+1
        while end<len(scores) and scores[order[end]]==scores[order[start]]: end+=1
        ranks[order[start:end]]=(start+1+end)/2; start=end
    return float((ranks[truth].sum()-positives*(positives+1)/2)/(positives*negatives))


def probability_metrics(y,p,reference=None):
    y,p=_inputs(y,p); k=p.shape[1]; target=np.eye(k)[y]; confidence=p.max(1); correct=p.argmax(1)==y; ece=0.
    for low,high in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
        use=(confidence>low)&(confidence<=high)
        if use.any(): ece+=use.mean()*abs(correct[use].mean()-confidence[use].mean())
    brier=float(np.mean(np.sum((p-target)**2,axis=1))); aucs=[binary_auc(y==c,p[:,c]) for c in range(k)]
    result={'nll':float(-np.mean(np.log(np.maximum(p[np.arange(len(y)),y],1e-15)))),'brier':brier,
            'ece_10_bins':float(ece),'macro_ovr_auc':None if any(a is None for a in aucs) else float(np.mean(aucs))}
    if reference is not None:
        reference=np.asarray(reference,dtype=float)
        if reference.shape!=(k,) or (reference<0).any() or not np.isfinite(reference).all() or not np.isclose(reference.sum(),1):
            raise ValueError('Reference must be a class probability vector')
        ref_brier=np.mean(np.sum((reference[None,:]-target)**2,axis=1))
        result['brier_skill']=float(1-brier/ref_brier) if ref_brier>0 else None
    return result


def score(yb,yc,pb,pc):
    yb,pb=_inputs(yb,pb); yc,pc=_inputs(yc,pc)
    if pb.shape!=(len(yb),3) or pc.shape!=(len(yb),7): raise ValueError('Expected 3/7-class heads on identical rows')
    b=pb.argmax(1); c=pc.argmax(1); bm=confusion(yb,b,3); cm=confusion(yc,c,7)
    return {'n':len(b),'behaviour_accuracy':float(np.mean(yb==b)),'composition_accuracy':float(np.mean(yc==c)),
            'behaviour_macro_f1':macro_f1(bm),'composition_macro_f1':macro_f1(cm),
            'joint_accuracy':float(np.mean((yb==b)&(yc==c))),'behaviour_confusion':bm.tolist(),'composition_confusion':cm.tolist()}


def continuous_metrics(truth,estimate):
    truth=np.asarray(truth,dtype=float); estimate=np.asarray(estimate,dtype=float)
    if truth.ndim!=1 or not len(truth) or truth.shape!=estimate.shape or not np.isfinite([truth,estimate]).all():
        raise ValueError('Expected finite matching continuous targets')
    residual=estimate-truth; mse=float(np.mean(residual**2)); rmse=float(np.sqrt(mse)); total=float(np.sum((truth-truth.mean())**2))
    r2=float(1-np.sum(residual**2)/total) if total>0 else (1. if mse==0 else 0.)
    return {'slope_MAE':float(np.mean(abs(residual))),'slope_MSE':mse,'slope_RMSE':rmse,'slope_R2':r2,
            'slope_RPD':float(np.std(truth,ddof=1)/rmse) if len(truth)>1 and rmse>0 and total>0 else None,
            'MAPE':'not reported for targets containing zero; never applied to class codes'}


def group_bootstrap_difference(group,correct_a,correct_b,seed=19,repeats=1500):
    group=np.asarray(group); a=np.asarray(correct_a,dtype=float); b=np.asarray(correct_b,dtype=float)
    if group.ndim!=1 or not len(group) or a.shape!=group.shape or b.shape!=group.shape or not np.isfinite([a,b]).all():
        raise ValueError('Paired correctness values must match nonempty physical groups')
    ids=np.unique(group); deltas=np.array([np.mean(a[group==g]-b[group==g]) for g in ids]); rng=np.random.default_rng(seed)
    samples=np.mean(deltas[rng.integers(0,len(ids),(repeats,len(ids)))],axis=1)
    null=np.mean(deltas[None,:]*rng.choice([-1,1],size=(5000,len(ids))),axis=1)
    return {'difference':float(deltas.mean()),'pooled_joint_difference':float(np.mean(a-b)),
            'ci95':np.quantile(samples,[.025,.975]).tolist(),
            'paired_group_permutation_p':float((1+np.sum(abs(null)>=abs(deltas.mean())))/(1+len(null))),
            'independent_groups':len(ids),'unit':'parent physical branch, averaging all its supported modes'}
