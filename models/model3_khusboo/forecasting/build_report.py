"""Charts, Excel source data and a simple explanation of the revised task."""
import gzip
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from forecast_core import ROOT,BUILDINGS
from train_forecasts import write_json,regression_scores
sys.path.append(str(ROOT.parent/'.venv/packages'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=ROOT/'results';O=ROOT/'outputs'
def load(name):return json.loads((R/name).read_text())
def records(frame):return json.loads(frame.to_json(orient='records',date_format='iso'))
def figure(name):
    plt.tight_layout()
    if '--report-only' not in sys.argv:plt.savefig(O/f'{name}.png',dpi=170,bbox_inches='tight')
    plt.close()

def main():
    O.mkdir(exist_ok=True)
    h=load('hourly_test_metrics.json');m=load('monthly_test_metrics.json')
    hp=pd.read_csv(R/'hourly_test_predictions.csv.gz',parse_dates=['timestamp','origin'])
    mp=pd.read_csv(R/'monthly_test_predictions.csv.gz',parse_dates=['origin'])
    fh=pd.read_csv(R/'hourly_future_forecasts.csv',parse_dates=['origin','timestamp'])
    fm=pd.read_csv(R/'monthly_future_forecasts.csv',parse_dates=['origin'])
    maxima=fh.loc[fh.groupby('building').forecast_kw.idxmax(),['building','timestamp','forecast_kw','peak_threshold_kw','predicted_peak']]
    maxima.to_csv(R/'daily_maximum_forecasts.csv',index=False,float_format='%.12g')
    dictionary=[]
    for task in ('hourly','monthly'):
        for col in load(f'{task}_selection.json')['features']:
            if col.startswith('lag_'):definition='Same target-hour power this many complete days earlier; available before origin midnight'
            elif col.startswith('previous_day_'):definition='Statistic of the last completed local day; needs at least 20 valid hours'
            elif col.startswith('previous_') and 'd_mean' in col:definition='Mean of completed daily means before origin; at least half the rolling days (floor, minimum 2)'
            elif col.startswith('previous_month_'):definition='Mean kW of this many completed months before forecast origin; monthly coverage at least 80%'
            elif col=='previous_3m_mean':definition='Average of available means for the three completed preceding months'
            elif col=='same_month_last_year':definition='Mean kW in target month one calendar year earlier; known before every supported forecast origin'
            elif col.startswith('building_'):definition='Known meter/building indicator, supplied at forecast time'
            elif col=='horizon':definition='Requested month lead, 1 / 2 / 3, known at origin'
            else:definition='Known target-calendar cyclic sine/cosine or weekend indicator; no measured target-day data'
            unit='kW' if col.startswith(('lag_','previous_')) or col=='same_month_last_year' else ('months' if col=='horizon' else 'unitless')
            dictionary.append({'task':task,'feature':col,'unit':unit,'availability_definition':definition})
    pd.DataFrame(dictionary).to_csv(R/'feature_dictionary.csv',index=False)
    hourly=pd.read_csv(R/'hourly_measurements.csv.gz',parse_dates=['timestamp'])
    monthly=pd.read_csv(R/'monthly_measurements.csv')
    colors=['#284B7D','#CF7C35','#418474']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    example=hp[hp.building.eq('Academic')&hp.timestamp.lt('2017-01-08')].set_index('timestamp')[['actual_kw','predicted_kw','baseline_kw']].reindex(pd.date_range('2017-01-01',periods=168,freq='h')).rename_axis('timestamp').reset_index()
    example.to_csv(R/'chart_hourly_example.csv',index=False)
    fig,ax=plt.subplots(figsize=(11,4))
    for col,name,color in zip(['actual_kw','predicted_kw','baseline_kw'],['Actual','Day-ahead forecast','Last-week baseline'],colors):ax.plot(example.timestamp,example[col],label=name,color=color,lw=1.4)
    ax.set(ylabel='Hourly mean demand (kW)',title='Academic: first seven test days (day-ahead forecasts)');ax.legend(frameon=False,ncol=3)
    figure('hourly_test_example')
    fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True)
    one=mp[mp.lead_months.eq(1)].copy()
    one.to_csv(R/'chart_monthly_one_month_ahead.csv',index=False)
    for ax,b in zip(axes,BUILDINGS):
        g=one[one.building.eq(b)].set_index('target_month').reindex(pd.period_range('2017-01','2017-12',freq='M').astype(str))
        for col,name,color in zip(['actual_kw','predicted_kw','baseline_kw'],['Actual','One-month-ahead forecast','Same month last year'],colors):ax.plot(range(1,13),g[col],marker='o',ms=3,label=name,color=color)
        ax.set(ylabel=f'{b}\nkW',xticks=range(1,13));ax.legend(frameon=False,fontsize=8,ncol=3)
    axes[0].set_title('Reserved 2017 monthly demand; missing low-coverage months stay blank');axes[-1].set_xlabel('Target calendar month')
    figure('seasonal_test_forecasts')
    seasons=['Dec-Feb','Mar-May','Jun-Sep','Oct-Nov']
    seasonal=pd.DataFrame([{'season':s,'hourly_wape':h['by_season'][s]['wape'],'monthly_wape':m['by_season'][s]['wape'],'hourly_n':h['by_season'][s]['n'],'monthly_n':m['by_season'][s]['n']} for s in seasons])
    seasonal.to_csv(R/'chart_season_errors.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,4));x=np.arange(4)
    ax.bar(x-.18,seasonal.hourly_wape*100,.36,label='Day-ahead hourly',color=colors[0]);ax.bar(x+.18,seasonal.monthly_wape*100,.36,label='Monthly, leads 1–3',color=colors[1]);ax.set(xticks=x,xticklabels=seasons,ylabel='WAPE (%) — lower is better',title='Errors across explicit seasonal month groups');ax.legend(frameon=False)
    figure('seasonal_errors')
    cm=np.array(h['peak']['confusion_matrix'])
    fig,ax=plt.subplots(figsize=(6,4.5));ax.imshow(cm,cmap='Blues',vmin=0)
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Non-peak','Peak'],yticklabels=['Non-peak','Peak'],xlabel='Forecast label',ylabel='Actual label',title='2017 peak / non-peak detection')
    for i in range(2):
        for j in range(2):ax.text(j,i,f'{cm[i,j]:,}',ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black',fontsize=13)
    figure('peak_confusion_matrix')
    fig,ax=plt.subplots(figsize=(8,4))
    for b,c in zip(BUILDINGS,colors):
        g=fm[fm.building.eq(b)];ax.plot(g.target_month,g.forecast_kw,marker='o',color=c,label=b)
    ax.set(ylabel='Forecast monthly mean demand (kW)',title='Historical-origin forecast issued at 1 January 2018');ax.legend(frameon=False)
    figure('future_monthly_forecast')
    cvs=pd.concat([pd.read_csv(R/f'{t}_cv.csv') for t in ('hourly','monthly')],ignore_index=True)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,t in zip(axes,['hourly','monthly']):
        c=pd.read_csv(R/f'{t}_comparison.csv').iloc[::-1]
        ax.barh(c.model,c.mean_macro_wape*100,color=colors[0]);ax.set(title=t.capitalize()+' development selection',xlabel='Mean building-balanced WAPE (%)')
    figure('validation_comparison')
    baseline_peak=(hp.baseline_kw>hp.peak_threshold_kw).astype(int)
    peak_baselines={'last_week':classification_report(hp.actual_peak,baseline_peak,labels=[0,1],target_names=['non_peak','peak'],output_dict=True,zero_division=0),'always_non_peak':classification_report(hp.actual_peak,np.zeros(len(hp)),labels=[0,1],target_names=['non_peak','peak'],output_dict=True,zero_division=0)}
    write_json(R/'peak_baselines.json',peak_baselines)
    quality=hourly.assign(year=hourly.timestamp.dt.year).groupby(['building','year']).agg(total_hours=('power_kw','size'),eligible_hours=('power_kw','count')).reset_index()
    quality['excluded_hours']=quality.total_hours-quality.eligible_hours
    quality.to_csv(R/'hourly_quality_by_year.csv',index=False)
    thresholds=pd.DataFrame([{'building':b,'evaluation_threshold_kw':h['peak']['thresholds_kw'][b],'operational_threshold_kw':fh.loc[fh.building.eq(b),'peak_threshold_kw'].iloc[0]} for b in BUILDINGS])
    thresholds.to_csv(R/'peak_thresholds.csv',index=False)
    payload={'hourly_metrics':h,'monthly_metrics':m,'peak_baselines':peak_baselines,'hourly_predictions':records(hp[['timestamp','building','actual_kw','predicted_kw','baseline_kw','peak_threshold_kw','actual_peak','predicted_peak']]),'monthly_predictions':records(mp[['origin','target_month','building','lead_months','actual_kw','predicted_kw','baseline_kw']]),'future_hourly':records(fh),'future_monthly':records(fm),'seasons':records(seasonal),'cv':records(cvs),'example':records(example),'quality':records(quality),'monthly_quality':records(monthly),'thresholds':records(thresholds),'monthly_chart':records(one[['target_month','building','actual_kw','predicted_kw','baseline_kw']])}
    with gzip.open(R/'workbook_data.json.gz','wt',encoding='utf-8') as out:json.dump(payload,out,allow_nan=False)
    report=f'''# Model 3: seasonal demand forecasting and peak periods

This is the corrected task requested on 27 November 2026. Earlier building-recognition and circuit-feasibility artifacts are historical work; they are not the forecasts reported here.

## What the revised system predicts

For each known building (Academic, Facilities, Mess), it predicts **24 hourly average loads for the next day**, and the **average load in each of the next three months**. It flags a forecast hour as peak when forecast demand exceeds that building's historical 90th percentile. This is a data-defined high-demand rule, not a supplied tariff schedule, instantaneous surge threshold or equipment capacity limit. It does not assign R/I/RC/RLC circuit labels.

All forecasts are in kW, a measure of power. Monthly average kW is not monthly energy in kWh. Hourly averages can hide short minute-level spikes. A day-ahead forecast is issued at local midnight after the previous day's readings have arrived. A monthly forecast is issued at the start of a month after the preceding month has ended. These availability assumptions are explicit.

## Data and quality

The same user-supplied ZIP contains 6,331,530 minute measurements from three campus buildings, spanning 2013–2017. Unix seconds are converted to Asia/Kolkata before local-hour aggregation. Availability flags mask absent power readings. Invalid, nonfinite or negative power becomes missing. An hour needs at least 48 valid minutes (80%); a monthly target needs at least 80% of all expected hours. Missing targets are excluded rather than filled. Missing past inputs use a median learned only on the training period. Real zeros are retained. Current, voltage and power factor are unnecessary for the demand targets and are not predictors.

The source labels identify buildings and availability, not peak status or seasons. The four seasonal groups are explicitly defined as Dec–Feb, Mar–May, Jun–Sep and Oct–Nov; these are calendar groups rather than measured weather labels. Holidays/activity files are excluded because their advance publication is not documented. No weather forecasts were supplied.

## Features and model choice

The day-ahead model uses 20 inputs: matching hourly loads 1, 2, 7 and 14 days earlier; summaries of the last completed day; preceding 7/28-day daily means; cyclic hour, weekday and annual calendar variables; weekend status; and building indicators. It never uses actual demand from the day it is forecasting. Building identity is a legitimate known input for this demand forecast.

The monthly model uses 11 inputs: completed-month loads, the previous three-month average, the same target month last year, target-month seasonality, forecast horizon and known building. Longer-horizon targets from the same origin share the same completed historical months; actual intermediate future months are not injected. No forward interpolation or centered rolling windows are used.

The selected hourly model is **{h['selected']}**, a gradient-boosted tree regressor that combines small decision trees to predict load. The selected monthly model is **{m['selected']}**, ridge regression, which estimates a regularized relationship between past loads, seasonality and the future monthly mean. Each was compared against history-based baselines and the other learned model. Selection minimized mean validation WAPE equally weighted across the three buildings, so the largest building did not dominate selection. Settings were fixed before evaluating 2017. Negative predictions are clipped to zero, a predeclared physical bound.

## Chronological evaluation and leakage controls

Development validation blocks were 2015, January–June 2016 and July–December 2016, always trained on earlier completed targets. The final evaluation models were fitted using targets completed before **1 January 2017** and then frozen. All buildings share the same calendar boundaries. Every monthly training target must have finished before a validation/test origin, including the third-month horizon; boundary-crossing training labels are purged. The first 14 days of each hourly series and the first 12 calendar months of each monthly series provide historical context rather than evaluation samples.

Evaluation is rolling-origin: at each later midnight or month-start, historical observations already available at that origin may update the input features. Fitted evaluation model parameters never retrain during 2017. This is not a single forecast of all 2017 issued at the start of that year. The monthly test has **93 origin/building/horizon forecast records** covering **{mp[['building','target_month']].drop_duplicates().shape[0]} distinct building-month targets**; repeated target months at different horizons are not independent observations. There are **25,302 eligible building-hours** in the hourly test, out of 26,280 possible 2017 building-hours. Exclusions and split manifests are saved.

The ZIP and some 2017 data had already been inspected for the abandoned building-recognition experiment. Consequently this is a retrospective forecasting holdout, not a claim that 2017 had never been seen in any context. No 2017 forecasting scores were used for selection or post-test tuning. More recent unseen data would provide stronger external validation.

## Final 2017 results

| Forecast | MAE (kW) | RMSE (kW) | WAPE | Baseline MAE (kW) |
| --- | ---: | ---: | ---: | ---: |
| Next-day hourly | {h['overall']['mae_kw']:.3f} | {h['overall']['rmse_kw']:.3f} | {h['overall']['wape']:.2%} | {h['baseline_overall']['mae_kw']:.3f} |
| Monthly, horizons 1–3 | {m['overall']['mae_kw']:.3f} | {m['overall']['rmse_kw']:.3f} | {m['overall']['wape']:.2%} | {m['baseline_overall']['mae_kw']:.3f} |

MAE means average absolute difference between predicted and actual load. For example, 2.75 kW MAE means forecasts were about 2.75 kW away from the actual hourly load on average. WAPE is sum of absolute errors divided by sum of actual load across the scored observations. It is an error measure, not classification accuracy; **12.19% WAPE must not be presented as 87.81% accuracy**. R² is {h['overall']['r2']:.4f} for hourly forecasts and {m['overall']['r2']:.4f} for monthly forecasts.

The hourly baseline repeats the same hour one week earlier. The monthly baseline repeats the same month one year earlier. Missing baseline history falls back to that building's training median. Hourly MAE improves by {(1-h['overall']['mae_kw']/h['baseline_overall']['mae_kw']):.2%}. Monthly MAE improves by only {(1-m['overall']['mae_kw']/m['baseline_overall']['mae_kw']):.2%}; that is effectively tied overall, and the baseline has a lower monthly RMSE ({m['baseline_overall']['rmse_kw']:.3f} versus {m['overall']['rmse_kw']:.3f} kW). The monthly model must not be claimed to strongly outperform a seasonal baseline. Seasonal Dec–Feb monthly WAPE is {m['by_season']['Dec-Feb']['wape']:.2%}, showing that error varies substantially by season.

## Peak / non-peak evaluation

Peak thresholds fitted before 2017: Academic **{h['peak']['thresholds_kw']['Academic']:.2f} kW**, Facilities **{h['peak']['thresholds_kw']['Facilities']:.2f} kW**, Mess **{h['peak']['thresholds_kw']['Mess']:.2f} kW**. Peak flags are derived from the predicted demand; there is no independent classifier with inconsistent load and peak outputs. The 90th percentile is an explicit assumption because no professor-specified limit was supplied.

Accuracy: **{h['peak']['classification_report']['accuracy']:.2%}**. Peak precision: **{h['peak']['classification_report']['peak']['precision']:.2%}**. Peak recall: **{h['peak']['classification_report']['peak']['recall']:.2%}**. Peak F1: **{h['peak']['classification_report']['peak']['f1-score']:.4f}**. Macro F1: **{h['peak']['macro_f1']:.4f}**. The system missed **{cm[1,0]:,} of {cm[1].sum():,} actual peak hours** and raised **{cm[0,1]:,} false peak flags**. Predicting every hour as non-peak would already achieve {peak_baselines['always_non_peak']['accuracy']:.2%} accuracy, so recall and precision are essential alongside accuracy. The last-week baseline's peak F1 is {peak_baselines['last_week']['peak']['f1-score']:.4f}.

## Issued forecasts and their date

After freezing the evaluation results, separate operational versions were fitted on all eligible data through December 2017. These produce 72 hourly forecasts for **1 January 2018** and nine building-month forecasts for **January–March 2018**. Operational peak thresholds are recalculated using history through 2017 and saved separately from the evaluation thresholds. These forecasts are examples from the dataset's historical endpoint, **not forecasts for 2026**. Their future actual loads were not supplied, so no observed accuracy or predictive interval is claimed for them. The evaluated models and operational refits are separately named and serialized.

| Building | Jan 2018 average kW | Feb 2018 average kW | Mar 2018 average kW |
| --- | ---: | ---: | ---: |
'''
    for b in BUILDINGS:
        vals=fm[fm.building.eq(b)].sort_values('lead_months').forecast_kw.to_list()
        report+=f'| {b} | {vals[0]:.2f} | {vals[1]:.2f} | {vals[2]:.2f} |\n'
    report+='\nFor 1 January 2018, all 72 hourly forecasts are below their historical high-demand cutoffs, so all are classified non-peak. This does not mean the load is constant: the highest predicted hourly average within each day is still identifiable. It is a different meaning of “daily maximum” from exceeding a fixed historical peak threshold.\n\n| Building | Highest forecast hour (local) | Highest hourly mean (kW) | Historical peak cutoff (kW) |\n| --- | --- | ---: | ---: |\n'
    for row in maxima.itertuples():report+=f'| {row.building} | {row.timestamp:%Y-%m-%d %H:%M} | {row.forecast_kw:.2f} | {row.peak_threshold_kw:.2f} |\n'
    report+='''
## Figures and professor explanation

![Example next-day hourly forecast](hourly_test_example.png)
![Seasonal monthly forecasts](seasonal_test_forecasts.png)
![Seasonal error](seasonal_errors.png)
![Peak/non-peak confusion](peak_confusion_matrix.png)
![Historical-origin future monthly predictions](future_monthly_forecast.png)
![Development comparisons](validation_comparison.png)

“My revised Model 3 forecasts how much electricity each building will use tomorrow and over the next three months. It learns daily, weekly and yearly patterns from past power readings. It flags high-demand hours using a threshold defined from training history. I tested the forecasts on later dates, compared them with simple seasonal baselines, and reported forecast errors and missed peak hours.”

The main limits are three known meters at one campus, a short monthly history, missing measurement periods, no weather inputs, calendar-derived peak/season rules and changing operations. The model has not been shown to generalize to new buildings. Recent data is needed to make current forecasts. No claim is made about physical circuit types, utility tariff peak schedules, instantaneous safety limits or guaranteed detection of every peak. Saved checks verify causal feature construction by changing future data and checking that earlier forecasts' inputs do not change.

All test predictions, issued forecasts, quality records, split manifests, model-selection scores and chart source tables are saved in results/. Compressed CSVs can be opened with pandas directly or decompressed before loading in Excel. forecast_results.xlsx includes both prediction tables and all data behind its editable charts. Six causal-feature/metric tests and saved-model verification accompany this experiment.
'''
    (O/'report.md').write_text(report,encoding='utf-8')
    print('Saved forecasting report, six figures and Excel source data')

if __name__=='__main__':main()
