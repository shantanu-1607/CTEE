"""Frozen chronological selection and final 2017 evaluation of Model 3 forecasts."""
import argparse
import hashlib
import json
import platform
import zipfile
import time
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,classification_report,confusion_matrix,f1_score,balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from forecast_core import ROOT,BUILDINGS,SEED,HistoryBaseline,hourly_features,monthly_series,monthly_features,feature_columns

R=ROOT/'results'
A=ROOT/'artifacts'
def write_json(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False,default=str),encoding='utf-8')

def extract(archive):
    audit=json.loads((ROOT.parent/'results/dataset_audit.json').read_text())
    with archive.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
    if sha!=audit['archive_sha256']:raise ValueError('Archive differs from audited source')
    rows=[]
    with zipfile.ZipFile(archive) as z:
        status=pd.read_csv(z.open('data_present_status_buildings.csv'),usecols=['timestamp',*BUILDINGS])
        status.index=pd.DatetimeIndex(pd.to_datetime(status.pop('timestamp')))
        if not status.index.is_unique:raise ValueError('Duplicate presence timestamps')
        for filename,b in [('acad_build_mains.csv','Academic'),('facilities_build_mains.csv','Facilities'),('mess_build_mains.csv','Mess')]:
            f=pd.read_csv(z.open(filename),usecols=['timestamp','power'])
            if f.timestamp.duplicated().any():raise ValueError('Duplicate measurement timestamps')
            if f.timestamp.mod(60).ne(0).any():raise ValueError('Off-minute readings')
            # Store local timestamps without tz after explicitly converting Unix UTC.
            f.index=pd.DatetimeIndex(pd.to_datetime(f.timestamp,unit='s',utc=True)).tz_convert('Asia/Kolkata').tz_localize(None)
            s=f.power.where(np.isfinite(f.power)&f.power.ge(0)&status[b].reindex(f.index).eq(1))/1000
            hourly=s.resample('h').agg(['mean','count'])
            hourly['power_kw']=hourly['mean'].where(hourly['count']>=48)
            hourly=hourly.assign(timestamp=hourly.index,building=b).rename(columns={'count':'valid_minutes'})
            rows.append(hourly[['timestamp','building','power_kw','valid_minutes']].reset_index(drop=True))
            print('Aggregated',b,len(hourly),flush=True)
    hourly=pd.concat(rows,ignore_index=True)
    hourly.to_csv(R/'hourly_measurements.csv.gz',index=False,compression='gzip',float_format='%.12g')
    return hourly

def regression_scores(frame,pred):
    y=frame.actual_kw.to_numpy()
    mae=float(mean_absolute_error(y,pred))
    total=float(np.abs(y).sum())
    wape=float(np.abs(y-pred).sum()/total) if total>0 else None
    return {'mae_kw':mae,'rmse_kw':float(np.sqrt(mean_squared_error(y,pred))),'wape':wape,'r2':float(r2_score(y,pred)) if len(y)>1 else None,'n':len(y)}

def macro_wape(frame,pred):
    return float(np.mean([regression_scores(frame.loc[frame.building.eq(b)],np.asarray(pred)[frame.building.eq(b)])['wape'] for b in BUILDINGS]))

def candidates(task):
    imputer=lambda:SimpleImputer(strategy='median',keep_empty_features=True)
    if task=='hourly':
        baselines={'same_hour_last_week':HistoryBaseline(('lag_7d',)),'day_week_blend':HistoryBaseline(('lag_1d','lag_7d'))}
        ridge_alpha=100
    else:
        baselines={'last_completed_month':HistoryBaseline(('previous_month_1',)),'same_month_last_year':HistoryBaseline(('same_month_last_year',))}
        ridge_alpha=10
    return {**baselines,'ridge':make_pipeline(imputer(),StandardScaler(),Ridge(alpha=ridge_alpha)), 'gradient_boosting':make_pipeline(imputer(),HistGradientBoostingRegressor(max_iter=160,max_leaf_nodes=15 if task=='hourly' else 7,min_samples_leaf=30 if task=='hourly' else 10,l2_regularization=10,early_stopping=False,random_state=SEED))}

def folds_for(frame,task):
    bounds=[('2015-01-01','2016-01-01'),('2016-01-01','2016-07-01'),('2016-07-01','2017-01-01')]
    for i,(start,end) in enumerate(bounds):
        start,end=pd.Timestamp(start),pd.Timestamp(end)
        if task=='hourly':
            train=frame[frame.timestamp<start]
            valid=frame[(frame.origin>=start)&(frame.timestamp<end)]
        else:
            # Purge ALL targets that finish after forecast availability cutoff.
            train=frame[frame.target_end<start]
            valid=frame[(frame.origin>=start)&(frame.origin<end)&(frame.target_end<end)]
        yield i+1,train.reset_index(drop=True),valid.reset_index(drop=True)

def run_task(frame,task):
    features=feature_columns(frame)
    observed=frame[frame.actual_kw.notna()].copy()
    models=candidates(task)
    cv=[]
    assignments=[]
    for i,tr,va in folds_for(observed,task):
        assert len(tr)>0 and len(va)>0
        for role,f in [('train',tr),('validation',va)]:
            for b,g in f.groupby('building'):
                assignments.append({'task':task,'fold':i,'role':role,'building':b,'n':len(g),'origin_start':str(g.origin.min()),'origin_end':str(g.origin.max()),'latest_target':str(g.timestamp.max() if task=='hourly' else g.target_end.max())})
        for name,model in models.items():
            start=time.perf_counter()
            est=clone(model).fit(tr[features],tr.actual_kw)
            pred=np.maximum(est.predict(va[features]),0)
            cv.append({'task':task,'fold':i,'model':name,'macro_building_wape':macro_wape(va,pred),**regression_scores(va,pred),'seconds':time.perf_counter()-start})
            print(task,i,name,cv[-1]['macro_building_wape'],flush=True)
    comparison=pd.DataFrame(cv).groupby('model',sort=False).agg(mean_macro_wape=('macro_building_wape','mean'),sd_macro_wape=('macro_building_wape','std')).reset_index().sort_values('mean_macro_wape',kind='stable')
    selected=str(comparison.iloc[0].model)
    comparison.to_csv(R/f'{task}_comparison.csv',index=False)
    pd.DataFrame(cv).to_csv(R/f'{task}_cv.csv',index=False)
    pd.DataFrame(assignments).to_csv(R/f'{task}_folds.csv',index=False)
    cutoff=pd.Timestamp('2017-01-01')
    if task=='hourly':
        dev=observed[observed.timestamp<cutoff]
        test=observed[observed.origin>=cutoff]
    else:
        dev=observed[observed.target_end<cutoff]
        test=observed[(observed.origin>=cutoff)&(observed.target_end<pd.Timestamp('2018-01-01'))]
    est=clone(models[selected]).fit(dev[features],dev.actual_kw)
    bundle={'model':est,'features':features,'task':task,'selected':selected,'fit_cutoff':'2017-01-01','clip_predictions_min_kw':0,'forecast_mode':'fixed fitted parameters; observed history updates only before each origin'}
    if task=='hourly':
        # Thresholds use unique historical hourly targets, not feature rows from future.
        bundle['peak_threshold_kw']={b:float(g.actual_kw.quantile(.9)) for b,g in dev.groupby('building')}
    joblib.dump(bundle,A/f'{task}_evaluation.joblib',compress=3)
    joblib.dump(est[:-1] if hasattr(est,'steps') else {'fallback_by_building':est.by_building_},A/f'{task}_evaluation_preprocessing.joblib',compress=3)
    write_json(R/f'{task}_selection.json',{'selected':selected,'features':features,'validation_mean_macro_building_wape':float(comparison.iloc[0].mean_macro_wape),'fit_cutoff':'2017-01-01','development_rows':len(dev),'test_rows':len(test)})
    # Selection and serialization completed before any final test score.
    result=test[[c for c in frame if c not in features]].copy().reset_index(drop=True)
    result['predicted_kw']=np.maximum(est.predict(test[features]),0)
    baseline_name=next(iter(models)) if task=='hourly' else 'same_month_last_year'
    baseline=clone(models[baseline_name]).fit(dev[features],dev.actual_kw)
    result['baseline_kw']=baseline.predict(test[features])
    metrics={'selected':selected,'overall':regression_scores(test,result.predicted_kw),'baseline_name':baseline_name,'baseline_overall':regression_scores(test,result.baseline_kw),'macro_building_wape':macro_wape(test,result.predicted_kw),'by_building':{},'by_season':{}}
    for b,g in result.groupby('building'):
        metrics['by_building'][b]=regression_scores(g,g.predicted_kw)
    for s,g in result.groupby('season'):
        metrics['by_season'][s]=regression_scores(g,g.predicted_kw)
    if task=='monthly':
        metrics['by_lead_months']={str(h):regression_scores(g,g.predicted_kw) for h,g in result.groupby('lead_months')}
    else:
        result['peak_threshold_kw']=result.building.map(bundle['peak_threshold_kw'])
        result['actual_peak']=(result.actual_kw>result.peak_threshold_kw).astype(int)
        result['predicted_peak']=(result.predicted_kw>result.peak_threshold_kw).astype(int)
        metrics['peak']={'rule':'strictly greater than development 90th percentile per building','thresholds_kw':bundle['peak_threshold_kw'],'confusion_matrix':confusion_matrix(result.actual_peak,result.predicted_peak,labels=[0,1]).tolist(),'classification_report':classification_report(result.actual_peak,result.predicted_peak,labels=[0,1],target_names=['non_peak','peak'],output_dict=True,zero_division=0),'macro_f1':float(f1_score(result.actual_peak,result.predicted_peak,average='macro',zero_division=0)),'balanced_accuracy':float(balanced_accuracy_score(result.actual_peak,result.predicted_peak))}
        metrics['peak']['by_building']={b:classification_report(g.actual_peak,g.predicted_peak,labels=[0,1],target_names=['non_peak','peak'],output_dict=True,zero_division=0) for b,g in result.groupby('building')}
    result.to_csv(R/f'{task}_test_predictions.csv.gz',index=False,compression='gzip',float_format='%.12g')
    write_json(R/f'{task}_test_metrics.json',metrics)
    return bundle,result

def final_future(hourly,monthly):
    """Separate operational refit AFTER frozen test; never attach its training score."""
    forecasts={}
    all_models={}
    for task in ('hourly','monthly'):
        evaluation=joblib.load(A/f'{task}_evaluation.joblib')
        model=clone(evaluation['model'])
        if task=='hourly':
            origin=hourly.timestamp.max().normalize()+pd.Timedelta(days=1)
            future=pd.DataFrame([{'timestamp':t,'building':b,'power_kw':np.nan,'valid_minutes':0} for b in BUILDINGS for t in pd.date_range(origin,periods=24,freq='h')])
            data=hourly_features(pd.concat([hourly,future],ignore_index=True))
            train=data[data.actual_kw.notna()&(data.timestamp<origin)]
            targets=data[data.origin.eq(origin)].copy()
            thresholds={b:float(g.actual_kw.quantile(.9)) for b,g in train.groupby('building')}
        else:
            data=monthly_features(monthly,future=True)
            origin=pd.Timestamp('2018-01-01')
            train=data[data.actual_kw.notna()&(data.target_end<origin)]
            targets=data[data.origin.eq(origin)].copy()
        features=evaluation['features']
        model.fit(train[features],train.actual_kw)
        bundle={**evaluation,'model':model,'fit_cutoff':str(origin),'status':'post-evaluation refit on all observations through 2017; NOT the evaluated model'}
        if task=='hourly':bundle['peak_threshold_kw']=thresholds
        joblib.dump(bundle,A/f'{task}_operational.joblib',compress=3)
        joblib.dump(model[:-1] if hasattr(model,'steps') else {'fallback_by_building':model.by_building_},A/f'{task}_operational_preprocessing.joblib',compress=3)
        cols=['origin','building']+(['timestamp'] if task=='hourly' else ['target_month','lead_months'])
        pred=targets[cols].copy()
        pred['forecast_kw']=np.maximum(model.predict(targets[features]),0)
        if task=='hourly':
            pred['peak_threshold_kw']=pred.building.map(thresholds)
            pred['predicted_peak']=(pred.forecast_kw>pred.peak_threshold_kw).astype(int)
        pred.to_csv(R/f'{task}_future_forecasts.csv',index=False,float_format='%.12g')
        forecasts[task]=pred
    return forecasts

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('zip',type=__import__('pathlib').Path)
    parser.add_argument('--reuse-hourly',action='store_true')
    args=parser.parse_args()
    R.mkdir(exist_ok=True);A.mkdir(exist_ok=True)
    if any((R/f'{t}_test_metrics.json').exists() for t in ('hourly','monthly')):raise RuntimeError('Frozen final test already evaluated; reproduce in a fresh experiment copy')
    hourly=pd.read_csv(R/'hourly_measurements.csv.gz',parse_dates=['timestamp']) if args.reuse_hourly else extract(args.zip)
    monthly=monthly_series(hourly)
    monthly.to_csv(R/'monthly_measurements.csv',index=False,float_format='%.12g')
    protocol={'seed':SEED,'timezone':'Asia/Kolkata','hourly_target':'next local day, 24 separate hourly mean-power forecasts at midnight','monthly_target':'average kW in each of next three calendar months from start-of-month origin','peak_rule':'hourly mean kW strictly above per-building development 90th percentile','season_groups':['Dec-Feb','Mar-May','Jun-Sep','Oct-Nov'],'season_note':'explicit month-based groups; no observed weather or supplied season labels','minimum_hour_minutes':48,'minimum_month_coverage':.8,'test':'2017, reserved by timestamp/target completion; all buildings share dates','folds':'2015, 2016 H1, 2016 H2; expanding training; monthly training labels must finish before origin','selection':'minimum mean macro-building WAPE across chronological development folds','known_inputs':'past power and future calendar only; no future activity label, future measurements, weather or measured target-day summaries','deployment':'evaluation artifacts frozen; separate operational refits use data through 2017 and issue January 2018 forecasts','source_sha256':json.loads((ROOT.parent/'results/dataset_audit.json').read_text())['archive_sha256'],'versions':{'python':platform.python_version(),'sklearn':sklearn.__version__,'pandas':pd.__version__,'numpy':np.__version__}}
    write_json(R/'protocol.json',protocol)
    with threadpool_limits(limits=2):
        run_task(hourly_features(hourly),'hourly')
        run_task(monthly_features(monthly),'monthly')
        final_future(hourly,monthly)
    print('Frozen evaluation and separate future forecast models saved',flush=True)

if __name__=='__main__':main()
