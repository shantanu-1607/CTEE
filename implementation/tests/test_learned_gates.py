"""Finite-difference checks of the new trainable evidence-gating weights."""
import numpy as np
import pytest
from ctee.scratch import NetworkConfig,TwoHeadMLP,Adam


def test_gated_backpropagation_checks_every_parameter():
    model=TwoHeadMLP(NetworkConfig(4,(3,),activation='tanh',dropout=0.,seed=7,
                        prior_strength=1.3,learned_physics_gate=True,gate_feature_indices=(0,2)))
    rng=np.random.default_rng(23);x=rng.normal(size=(5,4));yb=np.array([0,1,2,0,1]);yc=np.array([0,1,6,4,3])
    model.params['W_gate_behaviour'][:]=[[.2],[-.3]]
    model.params['W_gate_composition'][:]=[[.1],[.4]]
    args=dict(priors={'behaviour':rng.normal(size=(5,3)),'composition':rng.normal(size=(5,7))},
              training=False,physics_weight=.1,slopes=np.array([0.,-1.,1.,0.,-1.]),l2=.007)
    _,gradients=model.loss_and_gradients(x,yb,yc,**args)
    for name,parameter in model.params.items():
        numerical=np.zeros_like(parameter)
        for index in np.ndindex(parameter.shape):
            original=parameter[index];parameter[index]=original+1e-6
            plus=model.loss_and_gradients(x,yb,yc,**args)[0]['loss']
            parameter[index]=original-1e-6;minus=model.loss_and_gradients(x,yb,yc,**args)[0]['loss']
            parameter[index]=original;numerical[index]=(plus-minus)/2e-6
        np.testing.assert_allclose(gradients[name],numerical,atol=2e-7,rtol=2e-5,err_msg=name)


def test_gates_learn_to_reject_wrong_physics_and_preserve_checkpoint(tmp_path):
    model=TwoHeadMLP(NetworkConfig(2,(2,),dropout=0.,prior_strength=2.,learned_physics_gate=True))
    for name,p in model.params.items():p[:]=0.
    x=np.array([[1.,0.],[-1.,0.]]);yb=np.array([0,1]);yc=np.array([0,1])
    priors={'behaviour':np.array([[-4.,0.,0.],[0.,-4.,0.]]),
            'composition':np.array([[-4.,0.,0.,0.,0.,0.,0.],[0.,-4.,0.,0.,0.,0.,0.]])}
    before=model.physics_gate_values(x)['composition'].copy();optimizer=Adam(model.params,learning_rate=.05)
    for _ in range(10):
        _,grad=model.loss_and_gradients(x,yb,yc,priors=priors);optimizer.step(grad)
    assert np.all(model.physics_gate_values(x)['composition']<before)
    model.save(tmp_path/'gated.npz');restored=TwoHeadMLP.load(tmp_path/'gated.npz')
    np.testing.assert_array_equal(restored.forward(x,priors)['composition'],model.forward(x,priors)['composition'])


def test_gate_reads_only_declared_quality_features():
    model=TwoHeadMLP(NetworkConfig(4,(3,),prior_strength=1.,learned_physics_gate=True,gate_feature_indices=(1,3)))
    model.params['W_gate_composition'][:]=[[1.],[2.]]
    x=np.ones((2,4));before=model.physics_gate_values(x)['composition'];x[:,[0,2]]=1000.
    np.testing.assert_array_equal(before,model.physics_gate_values(x)['composition'])
    with pytest.raises(ValueError,match='prior logits'):model.forward(x)
