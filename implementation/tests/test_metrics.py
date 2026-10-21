import numpy as np
import pytest
from ctee.metrics import binary_auc,confusion,macro_f1,probability_metrics,continuous_metrics,score

def test_confusion_and_macro_f1_by_hand():
    m=confusion([0,0,1,1],[0,1,1,1],2)
    np.testing.assert_array_equal(m,[[1,1],[0,2]])
    assert macro_f1(m)==pytest.approx((2/3+.8)/2)

def test_tied_auc_and_missing_class():
    assert binary_auc([0,1,0,1],[.1,.5,.5,.9])==.875
    assert binary_auc([1,0],[.1,.9])==0.
    assert binary_auc([1,1],[.1,.9]) is None

def test_probability_metrics_by_hand():
    r=probability_metrics(np.array([0,1]),np.array([[.75,.25],[.25,.75]]),[.5,.5])
    assert r['nll']==pytest.approx(-np.log(.75))
    assert r['brier']==.125 and r['ece_10_bins']==.25 and r['brier_skill']==.75
    assert r['macro_ovr_auc']==1.

def test_undefined_auc_and_probability_validation():
    assert probability_metrics(np.array([0,0]),np.array([[1.,0],[1.,0]]))['macro_ovr_auc'] is None
    with pytest.raises(ValueError): probability_metrics(np.array([0]),np.array([[.8,.8]]))

def test_joint_accuracy_is_both_heads_not_average():
    yb=np.array([0,1,2]); yc=np.array([0,1,2])
    r=score(yb,yc,np.eye(3)[[0,1,0]],np.eye(7)[[0,2,2]])
    assert r['joint_accuracy']==1/3
    assert r['behaviour_accuracy']==r['composition_accuracy']==2/3

def test_continuous_metrics_use_actual_quantities():
    r=continuous_metrics([0.,1.,2.],[0.,2.,2.])
    assert r['slope_MAE']==pytest.approx(1/3) and r['slope_MSE']==pytest.approx(1/3)
    assert r['slope_R2']==.5
    assert continuous_metrics([1.,1.],[1.,1.])['slope_RPD'] is None
