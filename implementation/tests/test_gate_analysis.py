import numpy as np
from ctee.scratch import NetworkConfig,TwoHeadMLP
from ctee.features import N_FEATURES
from ctee.gate_analysis import analyze_gates


def test_counterfactual_does_not_modify_weights_and_zero_gate_logits_match():
    model=TwoHeadMLP(NetworkConfig(N_FEATURES,(4,),dropout=0.,prior_strength=1.,learned_physics_gate=True))
    x=np.zeros((6,N_FEATURES));yb=np.arange(6)%3;yc=np.arange(6)%7;groups=np.arange(6)
    before={k:v.copy() for k,v in model.params.items()}
    result=analyze_gates(model,x,yb,yc,np.full(6,'clean'),groups)['conditions']['clean']
    assert result['learned']==result['constant_half']
    assert result['gates']['composition']['mean']==.5
    assert result['paired_group_comparison']['difference']==0
    for k,v in before.items():np.testing.assert_array_equal(v,model.params[k])
