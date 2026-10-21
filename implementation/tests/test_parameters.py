import csv,io
import numpy as np
import pytest
from ctee.scratch import NetworkConfig,TwoHeadMLP,TrainingConfig,TrainingData,fit
from ctee.parameters import parameter_summary,export_parameters,preview_edit


def test_one_weight_edit_changes_predictions_without_mutating_selected_model():
    model=TwoHeadMLP(NetworkConfig(3,(4,),dropout=0.,seed=3));x=np.array([.1,.2,.3])
    original={k:v.copy() for k,v in model.params.items()}
    experiment,result=preview_edit(model,x,'b_composition',(0,),1.)
    for name,p in original.items():np.testing.assert_array_equal(model.params[name],p)
    assert result['edited']['composition'][0]>result['original']['composition'][0]
    assert experiment.params['b_composition'][0]==1.
    with pytest.raises(ValueError):preview_edit(model,x,'W0',(100,1),0.)


def test_parameter_export_is_complete_and_matches_arrays():
    model=TwoHeadMLP(NetworkConfig(2,(3,),dropout=0.))
    model.params['b_behaviour'][1]=.123
    rows=list(csv.DictReader(io.StringIO(export_parameters(model))))
    assert len(rows)==sum(p.size for p in model.params.values())
    assert next(r for r in rows if r['tensor']=='b_behaviour' and r['index']=='1')['learned_value']=='0.123'
    summary={r['tensor']:r for r in parameter_summary(model)}
    assert summary['b_behaviour']['changed_values']==1


def test_training_report_records_actual_learned_weights_and_gradient_updates():
    rng=np.random.default_rng(8);x=rng.normal(size=(24,3));data=TrainingData(x,np.arange(24)%3,np.arange(24)%7)
    model,report=fit(data,data,TrainingConfig(NetworkConfig(3,(4,),dropout=0.),epochs=3,batch_size=12))
    assert set(report['parameter_learning'])==set(model.params)
    assert all(r['change_l2']>0 and r['changed_values']>0 for r in report['parameter_learning'].values())
    assert all(row['mean_gradient_norm']>0 and row['parameter_update_l2']>0 for row in report['history'])
