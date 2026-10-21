from dataclasses import replace
import numpy as np
import pytest
from ctee.scratch import NetworkConfig,TrainingConfig,TrainingData,fit
from ctee.scratch.training import update_ema


def test_exponential_weight_averaging_matches_hand_updates():
    average={'W0':np.array([2.,-4.]),'b0':np.array([1.])}
    parameters={'W0':np.array([6.,0.]),'b0':np.array([3.])}
    update_ema(average,parameters,.75)
    np.testing.assert_array_equal(average['W0'],[3.,-3.]);np.testing.assert_array_equal(average['b0'],[1.5])
    update_ema(average,parameters,.75)
    np.testing.assert_array_equal(average['W0'],[3.75,-2.25])
    np.testing.assert_array_equal(parameters['W0'],[6.,0.])


def test_validation_selects_raw_or_averaged_weights_and_restores_best():
    x=np.random.default_rng(53).normal(size=(30,3));data=TrainingData(x,np.arange(30)%3,np.arange(30)%7)
    config=TrainingConfig(NetworkConfig(3,(5,),seed=1,dropout=0.),epochs=10,patience=10,batch_size=10,ema_decay=.8)
    model,history=fit(data,data,config)
    best=history['history'][history['best_epoch']-1]
    assert best['validation_loss']==min(best['raw_validation_loss'],best['averaged_validation_loss'])
    assert history['weight_source']==best['weight_source']
    restored,_=fit(data,data,replace(config,epochs=history['best_epoch']))
    for name,p in model.params.items():np.testing.assert_array_equal(p,restored.params[name])
    with pytest.raises(ValueError):replace(config,ema_decay=1.)
