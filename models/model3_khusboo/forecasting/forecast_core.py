"""Causal feature construction for day-ahead and one-to-three-month forecasts."""
from pathlib import Path
import os
# Bound automatic CPU discovery on Windows; avoids invoking obsolete WMIC.
os.environ.setdefault('LOKY_MAX_CPU_COUNT','2')
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

ROOT = Path(__file__).resolve().parent
BUILDINGS = ['Academic', 'Facilities', 'Mess']
SEED = 20261127

def season(month):
    # Explicit month groups; no weather or measured season labels are supplied.
    return 'Dec-Feb' if month in (12,1,2) else 'Mar-May' if month in (3,4,5) else 'Jun-Sep' if month in (6,7,8,9) else 'Oct-Nov'

def calendar_features(index):
    return pd.DataFrame({'hour_sin':np.sin(2*np.pi*index.hour/24), 'hour_cos':np.cos(2*np.pi*index.hour/24), 'dow_sin':np.sin(2*np.pi*index.dayofweek/7), 'dow_cos':np.cos(2*np.pi*index.dayofweek/7), 'year_sin':np.sin(2*np.pi*(index.dayofyear-1)/365.25), 'year_cos':np.cos(2*np.pi*(index.dayofyear-1)/365.25), 'weekend':(index.dayofweek>=5).astype(int)},index=index)

def hourly_features(hourly):
    frames=[]
    for b, frame in hourly.groupby('building',sort=False):
        s=frame.set_index('timestamp').power_kw.sort_index()
        idx=pd.date_range(s.index.min().normalize(),s.index.max().normalize()+pd.Timedelta(hours=23),freq='h')
        s=s.reindex(idx)
        f=calendar_features(idx)
        for d in (1,2,7,14): f[f'lag_{d}d']=s.shift(24*d)
        days=s.resample('D').agg(['mean','std','max','min','count'])
        # All daily-history features stop BEFORE the origin midnight, regardless of target hour.
        valid=days['count']>=20
        for col in ('mean','std','max','min'):
            past=days[col].where(valid).shift(1)
            f[f'previous_day_{col}']=past.reindex(idx.normalize()).to_numpy()
        for d in (7,28):
            past=days['mean'].where(valid).shift(1).rolling(d,min_periods=max(2,d//2)).mean()
            f[f'previous_{d}d_mean']=past.reindex(idx.normalize()).to_numpy()
        for name in BUILDINGS: f[f'building_{name}']=int(b==name)
        f['actual_kw']=s
        f['timestamp']=idx
        f['origin']=idx.normalize()
        f['history_end']=idx.normalize()-pd.Timedelta(hours=1)
        f['building']=b
        f['season']=[season(m) for m in idx.month]
        # Require two weeks elapsed; preserve missing targets for future inference.
        frames.append(f.loc[idx>=idx.min()+pd.Timedelta(days=14)].reset_index(drop=True))
    return pd.concat(frames,ignore_index=True)

def monthly_series(hourly):
    rows=[]
    for b,f in hourly.groupby('building'):
        s=f.set_index('timestamp').power_kw
        for month,g in s.groupby(s.index.to_period('M')):
            expected=month.days_in_month*24
            coverage=g.notna().sum()/expected
            rows.append({'building':b,'month':str(month),'mean_kw':g.mean() if coverage>=.8 else np.nan,'valid_hours':int(g.notna().sum()),'expected_hours':expected,'coverage':coverage})
    return pd.DataFrame(rows)

def monthly_features(monthly, future=False):
    rows=[]
    for b,frame in monthly.groupby('building'):
        s=frame.assign(period=pd.PeriodIndex(frame.month,freq='M')).set_index('period').mean_kw
        end=s.index.max()+1 if future else s.index.max()
        for origin in pd.period_range(s.index.min()+12,end,freq='M'):
            for lead in (1,2,3):
                target=origin+lead-1
                actual=s.get(target,np.nan)
                if not future and target>s.index.max():continue
                row={'origin':origin.start_time,'target_month':str(target),'target_end':target.end_time,'history_end':(origin-1).end_time,'building':b,'lead_months':lead,'actual_kw':actual,'season':season(target.month), 'month_sin':np.sin(2*np.pi*target.month/12),'month_cos':np.cos(2*np.pi*target.month/12),'horizon':lead}
                for lag in (1,2,3):row[f'previous_month_{lag}']=s.get(origin-lag,np.nan)
                row['same_month_last_year']=s.get(target-12,np.nan)
                row['previous_3m_mean']=s.reindex(pd.period_range(origin-3,origin-1,freq='M')).mean()
                for name in BUILDINGS:row[f'building_{name}']=int(b==name)
                rows.append(row)
    return pd.DataFrame(rows)

META={'actual_kw','timestamp','origin','history_end','building','season','target_month','target_end','lead_months'}
def feature_columns(frame):return [c for c in frame if c not in META]

class HistoryBaseline(RegressorMixin, BaseEstimator):
    def __init__(self,columns=('lag_7d',)):
        self.columns=columns
    def fit(self,X,y):
        self.fallback_=float(np.median(y))
        self.by_building_={b:float(np.median(np.asarray(y)[X[f'building_{b}'].eq(1).to_numpy()])) for b in BUILDINGS}
        return self
    def predict(self,X):
        # Row-wise mean for a two-history baseline; no target-period observations.
        values=X[list(self.columns)].mean(axis=1)
        for b in BUILDINGS:values=values.where(~(X[f'building_{b}'].eq(1)&values.isna()),self.by_building_[b])
        return values.fillna(self.fallback_).clip(lower=0).to_numpy()
