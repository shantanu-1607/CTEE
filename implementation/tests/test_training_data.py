import numpy as np
from ctee.features import N_BASE,N_FEATURES
from ctee.train import build_data

def test_training_adapter_reads_only_selected_rows_and_masks_evidence():
    x=np.zeros((5,N_FEATURES));x[:,0]=np.arange(5);x[:,N_BASE+3]=[1,0,1,0,1]
    d={'X':x,'yb':np.array([0,1,2,1,0]),'yc':np.array([1,2,3,4,5])}
    run={'enhanced':True,'seed':11,'prior_strength':.5}
    a=build_data(d,np.array([0,2]),run)
    d['X'][3:]=999
    b=build_data(d,np.array([0,2]),run)
    np.testing.assert_array_equal(a.x,b.x)
    np.testing.assert_array_equal(a.behaviour,[0,2])
    assert a.priors['composition'].shape==(2,7)
    c=build_data(d,np.array([0,1]),{'enhanced':False,'seed':11,'prior_strength':0.})
    assert c.x.shape==(2,N_BASE)
    np.testing.assert_array_equal(c.behaviour_mask,[True,False])
