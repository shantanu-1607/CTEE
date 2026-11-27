"""Tests prove feature causality by perturbing future measurements."""
import unittest
import numpy as np
import pandas as pd
from forecast_core import BUILDINGS,hourly_features,monthly_features,feature_columns,HistoryBaseline,season
from train_forecasts import folds_for,regression_scores

def hourly_fixture():
    idx=pd.date_range('2014-01-01',periods=24*80,freq='h')
    return pd.concat([pd.DataFrame({'timestamp':idx,'building':b,'power_kw':10+i+np.arange(len(idx))*.01,'valid_minutes':60}) for i,b in enumerate(BUILDINGS)],ignore_index=True)

class ForecastChecks(unittest.TestCase):
    def test_day_ahead_cannot_see_target_day(self):
        source=hourly_fixture()
        origin=pd.Timestamp('2014-02-20')
        a=hourly_features(source)
        altered=source.copy();altered.loc[altered.timestamp>=origin,'power_kw']=999999.
        b=hourly_features(altered)
        cols=feature_columns(a)
        np.testing.assert_allclose(a.loc[a.origin.eq(origin),cols],b.loc[b.origin.eq(origin),cols],equal_nan=True)
        self.assertTrue((a.loc[a.origin.eq(origin),'history_end']<origin).all())

    def test_daily_lags_and_midnight_history(self):
        source=hourly_fixture()
        f=hourly_features(source)
        row=f[(f.building=='Academic')&(f.timestamp==pd.Timestamp('2014-02-20 23:00'))].iloc[0]
        old=source[(source.building=='Academic')&(source.timestamp==pd.Timestamp('2014-02-19 23:00'))].power_kw.iloc[0]
        self.assertEqual(row.lag_1d,old)
        previous=source[(source.building=='Academic')&(source.timestamp>=pd.Timestamp('2014-02-19'))&(source.timestamp<pd.Timestamp('2014-02-20'))].power_kw.mean()
        self.assertAlmostEqual(row.previous_day_mean,previous)

    def test_monthly_causality_and_future_horizons(self):
        m=pd.concat([pd.DataFrame({'building':b,'month':pd.period_range('2013-01',periods=60,freq='M').astype(str),'mean_kw':np.arange(60)+10.}) for b in BUILDINGS],ignore_index=True)
        origin=pd.Timestamp('2016-01-01')
        a=monthly_features(m)
        changed=m.copy();changed.loc[changed.month>='2016-01','mean_kw']=999999.
        b=monthly_features(changed)
        np.testing.assert_allclose(a.loc[a.origin.eq(origin),feature_columns(a)],b.loc[b.origin.eq(origin),feature_columns(b)],equal_nan=True)
        future=monthly_features(m,future=True)
        self.assertEqual(len(future[future.origin.eq(pd.Timestamp('2018-01-01'))]),9)

    def test_monthly_fold_targets_finish_before_validation(self):
        m=pd.concat([pd.DataFrame({'building':b,'month':pd.period_range('2013-01',periods=60,freq='M').astype(str),'mean_kw':np.arange(60)+10.}) for b in BUILDINGS],ignore_index=True)
        for fold,tr,va in folds_for(monthly_features(m),'monthly'):
            self.assertLess(tr.target_end.max(),va.origin.min())
            self.assertTrue((va.history_end<va.origin).all())

    def test_baseline_missing_history_uses_only_training(self):
        X=pd.DataFrame({'lag_7d':[10,np.nan,20],**{f'building_{b}':[1,1,1] if b=='Academic' else [0,0,0] for b in BUILDINGS}})
        # Include all buildings in training, as the real protocol does.
        train=pd.concat([X.assign(**{f'building_{b}':int(b==k) for b in BUILDINGS}) for k in BUILDINGS],ignore_index=True)
        y=np.tile([2.,4.,6.],3)
        model=HistoryBaseline().fit(train,y)
        self.assertEqual(model.predict(X)[1],4)

    def test_error_units_and_season_mapping(self):
        f=pd.DataFrame({'actual_kw':[10.,20.]})
        score=regression_scores(f,np.array([8.,24.]))
        self.assertEqual(score['mae_kw'],3)
        self.assertAlmostEqual(score['wape'],.2)
        self.assertEqual([season(m) for m in (1,5,8,11)],['Dec-Feb','Mar-May','Jun-Sep','Oct-Nov'])

if __name__=='__main__':unittest.main()
