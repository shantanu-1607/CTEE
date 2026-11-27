"""Issue forecasts from saved parameters; never fit on future observations."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from forecast_core import ROOT,BUILDINGS,hourly_features,monthly_series,monthly_features

def predict(history,origin,task,bundle):
    if origin<pd.Timestamp(bundle['fit_cutoff']):raise ValueError('Origin predates the saved model training cutoff')
    if origin!=origin.normalize():raise ValueError('Origin must be local midnight')
    history=history.loc[history.timestamp<origin].copy()
    if set(history.building)!=set(BUILDINGS):raise ValueError('History must cover the three known buildings')
    if history.duplicated(['building','timestamp']).any():raise ValueError('Duplicate building-hour keys')
    if history.timestamp.dt.tz is not None:raise ValueError('Use timezone-naive Asia/Kolkata local timestamps')
    if (history.timestamp!=history.timestamp.dt.floor('h')).any():raise ValueError('History must already be aggregated to local hours')
    history['power_kw']=pd.to_numeric(history.power_kw,errors='coerce')
    history.loc[~np.isfinite(history.power_kw)|history.power_kw.lt(0),'power_kw']=np.nan
    if task=='hourly':
        for b,g in history.groupby('building'):
            recent=g.loc[g.timestamp>=origin-pd.Timedelta(days=14),'power_kw']
            if recent.notna().sum()<24:raise ValueError(f'Insufficient recent history for {b}')
        future=pd.DataFrame([{'timestamp':t,'building':b,'power_kw':np.nan} for b in BUILDINGS for t in pd.date_range(origin,periods=24,freq='h')])
        data=hourly_features(pd.concat([history,future],ignore_index=True))
        targets=data[data.origin.eq(origin)]
        cols=['origin','building','timestamp']
    else:
        if origin.day!=1:raise ValueError('Monthly forecast origin must be the start of a month')
        data=monthly_features(monthly_series(history),future=True)
        targets=data[data.origin.eq(origin)]
        cols=['origin','building','target_month','lead_months']
        if len(targets)!=9:raise ValueError('Monthly forecasting needs history through the preceding month and at least 12 months of calendar span')
    result=targets[cols].copy()
    result['forecast_kw']=np.maximum(bundle['model'].predict(targets[bundle['features']]),0)
    if task=='hourly':
        result['peak_threshold_kw']=result.building.map(bundle['peak_threshold_kw'])
        result['predicted_peak']=(result.forecast_kw>result.peak_threshold_kw).astype(int)
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--history',type=Path,required=True,help='CSV or CSV.gz: timestamp (local Asia/Kolkata), building, power_kw')
    parser.add_argument('--origin',required=True,help='Local midnight, YYYY-MM-DD')
    parser.add_argument('--task',choices=['hourly','monthly'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    history=pd.read_csv(args.history,parse_dates=['timestamp'])
    if not {'timestamp','building','power_kw'}.issubset(history):raise ValueError('Missing history columns')
    bundle=joblib.load(ROOT/'artifacts'/f'{args.task}_operational.joblib')
    result=predict(history,pd.Timestamp(args.origin),args.task,bundle)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(args.output,index=False,float_format='%.12g')
    print(f'Saved {len(result)} forecasts to {args.output}')

if __name__=='__main__':main()
