"""Reconcile saved forecasts, learned preprocessing and time boundaries; no refitting."""
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from forecast_core import ROOT,BUILDINGS,hourly_features,monthly_features
from train_forecasts import regression_scores,folds_for,write_json
from predict_forecasts import predict

def main():
    r=ROOT/'results';a=ROOT/'artifacts'
    hourly=pd.read_csv(r/'hourly_measurements.csv.gz',parse_dates=['timestamp'])
    monthly=pd.read_csv(r/'monthly_measurements.csv')
    checks=[]
    for task,frame in [('hourly',hourly_features(hourly)),('monthly',monthly_features(monthly))]:
        bundle=joblib.load(a/f'{task}_evaluation.joblib')
        observed=frame[frame.actual_kw.notna()].copy()
        cutoff=pd.Timestamp('2017-01-01')
        if task=='hourly':
            dev=observed[observed.timestamp<cutoff]
            test=observed[observed.origin>=cutoff]
        else:
            dev=observed[observed.target_end<cutoff]
            test=observed[(observed.origin>=cutoff)&(observed.target_end<pd.Timestamp('2018-01-01'))]
        assert (frame.history_end<frame.origin).all()
        for _,tr,va in folds_for(observed,task):
            assert (tr.timestamp.max() if task=='hourly' else tr.target_end.max())<va.origin.min()
            assert (va.timestamp.max() if task=='hourly' else va.target_end.max())<cutoff
        features=bundle['features']
        assert set(features).isdisjoint({'actual_kw','origin','history_end','timestamp','target_month','target_end'})
        estimator=bundle['model']
        if hasattr(estimator,'steps'):
            imputer=estimator.steps[0][1]
            medians=np.nanmedian(dev[features].to_numpy(),axis=0)
            medians=np.nan_to_num(medians,nan=0)
            np.testing.assert_allclose(imputer.statistics_,medians,rtol=1e-10,atol=1e-10)
        for mode in ('evaluation','operational'):
            saved=joblib.load(a/f'{task}_{mode}.joblib')['model']
            preprocessing=saved[:-1] if hasattr(saved,'steps') else {'fallback_by_building':saved.by_building_}
            joblib.dump(preprocessing,a/f'{task}_{mode}_preprocessing.joblib',compress=3)
        result=pd.read_csv(r/f'{task}_test_predictions.csv.gz')
        assert list(result.building)==list(test.building)
        np.testing.assert_allclose(result.actual_kw,test.actual_kw,rtol=1e-10)
        restored=np.maximum(estimator.predict(test[features]),0)
        np.testing.assert_allclose(result.predicted_kw,restored,rtol=1e-9)
        metrics=json.loads((r/f'{task}_test_metrics.json').read_text())
        for key,value in regression_scores(test,restored).items():
            if value is not None:assert abs(metrics['overall'][key]-value)<1e-9
        if task=='hourly':
            thresholds=dev.groupby('building').actual_kw.quantile(.9).to_dict()
            for b in BUILDINGS:assert abs(bundle['peak_threshold_kw'][b]-thresholds[b])<1e-10
            np.testing.assert_array_equal(result.actual_peak,(result.actual_kw>result.peak_threshold_kw).astype(int))
            np.testing.assert_array_equal(result.predicted_peak,(result.predicted_kw>result.peak_threshold_kw).astype(int))
            np.testing.assert_array_equal(confusion_matrix(result.actual_peak,result.predicted_peak,labels=[0,1]),metrics['peak']['confusion_matrix'])
        # Saved operational models must reproduce their future forecasts too.
        op=joblib.load(a/f'{task}_operational.joblib')
        future=predict(hourly,pd.Timestamp('2018-01-01'),task,op)
        old=pd.read_csv(r/f'{task}_future_forecasts.csv')
        np.testing.assert_allclose(old.forecast_kw,future.forecast_kw,rtol=1e-9)
        cols=['timestamp','origin','building','history_end'] if task=='hourly' else ['target_month','target_end','origin','building','history_end','lead_months']
        manifest=observed[cols].copy()
        manifest['partition']=np.where(observed.index.isin(dev.index),'development',np.where(observed.index.isin(test.index),'test','excluded_boundary'))
        manifest.to_csv(r/f'{task}_split_manifest.csv.gz',index=False,compression='gzip')
        checks.append({'task':task,'passed':True,'development_rows':len(dev),'test_rows':len(test),'features':len(features),'future_rows':len(future)})
    write_json(r/'verification.json',{'passed':True,'tasks':checks,'checks':['future perturbation unit tests separately verify causal features','all label end times precede fit/validation origins','all selection scores precede 2017','imputers equal development-only medians','saved models reproduce frozen test predictions and errors','peak thresholds match development-only 90th percentiles','peak labels and confusion matrix reconcile','operational inference reproduces issued forecasts'],'metrics_hashes':{task:hashlib.sha256((r/f'{task}_test_metrics.json').read_bytes()).hexdigest() for task in ('hourly','monthly')}})
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
