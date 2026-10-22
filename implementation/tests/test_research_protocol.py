import numpy as np
from ctee.features import N_FEATURES,N_BASE
from ctee.research import _train,GATE_FEATURES


def test_validation_search_never_uses_test_features_or_labels():
    rng=np.random.default_rng(18);n=40
    d={'X':rng.normal(size=(n,N_FEATURES)),'yb':np.arange(n)%3,'yc':np.arange(n)%7,
       'split':np.array(['train']*24+['validation']*12+['test']*4),
       'mask':np.full(n,31),'condition':np.full(n,'clean')}
    d['X'][:,N_BASE+3]=1.
    run={'variant':'test','enhanced':True,'robust':True,'prior_strength':0.,'physics_loss':0.}
    options={'hidden_widths':[4],'activation':'tanh','dropout':0.,'epochs':3,'patience':3,'batch_size':12,'ema_decay':.8}
    original,_,result=_train(d,run,options,11)
    d['X'][-4:]=np.nan;d['yb'][-4:]=100;d['yc'][-4:]=100
    changed,_,other=_train(d,run,options,11)
    assert result['balanced_validation_joint']==other['balanced_validation_joint']
    for k,p in original.params.items():np.testing.assert_array_equal(p,changed.params[k])
    assert len(set(GATE_FEATURES))==len(GATE_FEATURES)
