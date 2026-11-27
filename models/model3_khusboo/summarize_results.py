"""Produce figures, their flat source data and a plain-language experiment report."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
local_packages = ROOT / ".venv" / "packages"
if local_packages.exists(): sys.path.append(str(local_packages))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = ROOT / "results"
O = ROOT / "outputs" / "model3"
O.mkdir(parents=True, exist_ok=True)

def read(name): return json.loads((R / name).read_text(encoding="utf-8"))
def records(df): return json.loads(df.to_json(orient="records"))
def figure(name):
    plt.tight_layout()
    plt.savefig(O / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close()

def main():
    audit, protocol, selection, metrics = [read(n) for n in ["dataset_audit.json", "protocol.json", "selection.json", "test_metrics.json"]]
    daily = pd.read_csv(R / "daily_features.csv")
    manifest = pd.read_csv(R / "split_manifest.csv")
    cv = pd.read_csv(R / "cv_scores.csv")
    comparison = pd.read_csv(R / "model_comparison.csv")
    predictions = pd.read_csv(R / "test_predictions.csv")
    importance = pd.read_csv(R / "validation_feature_importance.csv")
    dictionary = pd.read_csv(R / "feature_dictionary.csv")
    quality = pd.read_csv(R / "day_quality.csv")
    counts = manifest.groupby(["label", "split"]).size().unstack(fill_value=0).reindex(protocol["labels"]).reset_index()
    counts.to_csv(R / "split_counts.csv", index=False)
    dev = daily[daily.sample_id.isin(manifest.loc[manifest.split.eq("development"), "sample_id"])]
    hourly = pd.DataFrame({"hour": range(24)})
    for label in protocol["labels"]:
        hourly[label] = [dev.loc[dev.label.eq(label), f"power_hour_{h:02d}"].mean() / 1000 for h in range(24)]
    hourly.to_csv(R / "development_hourly_power_kw.csv", index=False)
    predictions["month"] = predictions.date.str[:7]
    monthly = predictions.groupby("month").agg(total=("correct", "size"), correct=("correct", "sum")).reset_index()
    monthly["accuracy"] = monthly.correct / monthly.total
    monthly.to_csv(R / "test_monthly_accuracy.csv", index=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = ["#284B7D", "#CF7C35", "#418474"]
    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 4.6))
    ax.imshow(cm, cmap="Blues", vmin=0)
    ax.set(xticks=range(3), yticks=range(3), xticklabels=protocol["labels"], yticklabels=protocol["labels"], xlabel="Predicted building", ylabel="Actual building", title="Model 3: reserved future-day test")
    for i in range(3):
        for j in range(3): ax.text(j, i, str(cm[i,j]), ha="center", va="center", color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=13)
    figure("test_confusion_matrix")
    fig, ax = plt.subplots(figsize=(8,4.5))
    ax.barh(comparison.model.iloc[::-1], comparison.mean_macro_f1.iloc[::-1], xerr=comparison.sd_macro_f1.iloc[::-1], color=colors[0], capsize=3)
    ax.set(xlim=(0,1.05), xlabel="Mean validation macro F1 (bars: fold standard deviation)", title="Selection using earlier dates only")
    figure("model_comparison")
    fig, ax = plt.subplots(figsize=(8,4.5))
    for label,color in zip(protocol["labels"],colors): ax.plot(hourly.hour, hourly[label], label=label, color=color, lw=2)
    ax.set(xticks=range(0,24,2), xlabel="Local hour (Asia/Kolkata)", ylabel="Mean power (kW)", title="Daily load patterns in development data")
    ax.legend(frameon=False)
    figure("development_load_profiles")
    fig, ax = plt.subplots(figsize=(8,4.5))
    ax.bar(monthly.month, monthly.accuracy, color=colors[0])
    ax.set(ylim=(0,1.05), ylabel="Accuracy", xlabel="Reserved test month", title="Performance across the reserved test period")
    ax.tick_params(axis="x", rotation=45)
    figure("test_monthly_accuracy")
    top = importance.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8,5))
    ax.barh(top.feature, top.mean_f1_drop, xerr=top.sd_f1_drop, color=colors[2], capsize=2)
    ax.set(xlabel="Validation macro F1 drop after shuffling (5 repeats)", title="Feature importance on development validation data")
    figure("validation_feature_importance")
    payload = {"audit": audit, "protocol": protocol, "selection": selection, "metrics": metrics, "counts": records(counts), "cv": records(cv), "comparison": records(comparison), "predictions": records(predictions), "hourly": records(hourly), "monthly": records(monthly), "importance": records(importance.merge(dictionary, on="feature")), "quality": records(quality.groupby("label").agg(observed_days=("sample_id","size"), eligible_days=("eligible","sum")).reset_index())}
    (R / "workbook_data.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    raw_count = sum(v["rows"] for v in audit["measurements"].values())
    ci = metrics["approximate_week_block_bootstrap_95_ci"]
    report = f"""# Model 3: Khusboo building load-profile classifier

## The question this model answers

Given one completed day's electrical measurements, did it come from the **Academic**, **Facilities**, or **Mess** building in this dataset? These labels come from the three source filenames and the supplied Readme. They describe building identity. They are not R/I/RC/RLC, individual appliances, or confirmed physical resistive/inductive/capacitive load types.

There is only one building/meter for each class. The experiment measures recognition of future days from these same three meters. It cannot measure generalization to other buildings of the same type. Meter behavior may contribute to recognition.

## What was supplied

The archive has 9 CSVs, Readme.txt and .DS_Store: three electrical measurement files, five calendar files, and one building availability table. Its Readme describes a larger collection of 16 files; those additional measurement files were not supplied. No data was invented to replace them. The three measurement files contain **{raw_count:,} rows** in total, spanning 2013–2017. The detailed audit records dates, missingness, quantiles, sampling gaps and archive SHA-256.

Measurement timestamps are Unix seconds. Local days use **Asia/Kolkata (UTC+05:30)**, following the source's data description. The availability file uses local timestamp strings. The calendar has H/L activity and working-day labels, but these are separate annotations and are not used as predictors or as electrical load labels. Source documents were treated as evidence about the data, not as instructions to execute scripts or contact anyone.

## Cleaning and one example per day

One example is one building-day. Exact duplicate measurement rows collapse and conflicting duplicate timestamps fail explicitly. Invalid/negative power or power factors outside [-1,1] become missing. Availability flags only mask absent readings. A day needs at least 1,152 paired power/power-factor minutes (80% of 1,440) and at least 20 hours containing 30 paired minutes each. No interpolation bridges missing periods or dates. Missing hourly summaries remain missing for training-fitted median imputation. Real zero measurements are retained.

Of {len(quality):,} observed building-days, {int(quality.eligible.sum()):,} meet coverage rules; {int((~quality.eligible).sum()):,} fail. After duplicate/conflicting-profile checks, **{len(daily):,} examples** remain. Day quality and exclusions are saved individually. Identical daily signal fingerprints and rounded feature vectors are deduplicated before splitting; contradictory class labels are removed entirely. Earliest duplicate retention prevents copies spanning train/test. Days entirely absent from the source have no profile and are not counted as observed days.

## Why these 69 features

Power tells us how much electricity is being used. Power factor adds measured electrical behavior. Daily levels, spread, extremes and adjacent-minute changes summarize the signal. Twenty-four hourly means for each signal preserve how use changes through a day. Day/night and peak/mean ratios describe daily shape. Signed power factor is retained as supplied without claiming its sign proves any physical load category.

Current has substantial missingness and overlaps with power. Voltage and frequency largely describe the shared supply and may expose meter/site effects. They were audited but excluded. The model also excludes building names, filenames, dates/year, calendar labels, availability flags and coverage counts. Local hour is used only to align a daily profile. This is an end-of-day classifier, not a real-time or next-day forecast.

## How leakage was controlled

All three buildings on the same date stay in the same partition. The last 20% of eligible distinct dates are reserved for testing, starting **{protocol['test_start']}**. The preceding seven calendar days are excluded from development. Counts: **{len(dev):,} development**, **{sum(manifest.split.eq('embargo')):,} embargo**, **{len(predictions):,} final test** examples.

Three expanding chronological development folds train on earlier dates and validate on later ones, each with its own seven-day gap. There are no overlapping windows. Imputation and scaling are fitted within each training fold. Five fixed candidates were compared: a majority baseline, logistic regression, RBF SVM, random forest and histogram gradient boosting. The selection score is mean validation macro F1, giving each class equal weight. The complete settings and folds are saved. Hyperparameters were fixed before scoring, and no test-based tuning was performed. Aggregate file-quality inspection included the supplied archive; test performance was not used to design or choose models.

## Selected model and final results

Selected model: **{selection['selected_model']}**. Mean validation macro F1: **{selection['selection_macro_f1']:.4f}**. The winner was fitted on all development examples and serialized before final test scoring. Preprocessing is inside the saved pipeline and also saved separately. Test examples were not added to the fitted model.

An RBF support vector machine learns curved boundaries between the three groups of daily profiles. Standard scaling gives power in watts and power factor comparable influence on distances. Median imputation fills missing hourly summaries using development data only. A neural network was not needed: millions of minute readings become only a few thousand independent daily examples, and the simpler candidates can be compared directly with this amount of data. The SVM was chosen because it had the highest mean development validation macro F1; the small lead over logistic regression is not a claim of statistical significance.

| Final reserved test metric | Value |
| --- | ---: |
| Accuracy | {metrics['accuracy']:.2%} |
| Macro F1 | {metrics['macro_f1']:.4f} |
| Balanced accuracy | {metrics['balanced_accuracy']:.2%} |
| Correct / total | {metrics['correct']} / {metrics['n']} |
| Predeclared majority baseline accuracy | {metrics['majority_baseline']['accuracy']:.2%} |

Approximate 95% intervals from 1,000 bootstrap resamples of complete test calendar weeks: accuracy {ci['accuracy']['low']:.2%}–{ci['accuracy']['high']:.2%}; macro F1 {ci['macro_f1']['low']:.4f}–{ci['macro_f1']['high']:.4f}. Weekly resampling keeps buildings and nearby days together. These intervals describe this test period, not uncertainty across unseen campuses or meters. Weeks can still be correlated, and this approximation may understate longer-term uncertainty.

| Building | Precision | Recall | F1 | Test days |
| --- | ---: | ---: | ---: | ---: |
"""
    for label in protocol["labels"]:
        row = metrics["per_class"][label]
        report += f"| {label} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1-score']:.4f} | {int(row['support'])} |\n"
    report += f"""
\nValidation macro F1 ({selection['selection_macro_f1']:.4f}) exceeds final test macro F1 ({metrics['macro_f1']:.4f}). Performance declines across the held-out year: the first test month has {monthly.iloc[0].accuracy:.2%} accuracy, while the final month has {monthly.iloc[-1].accuracy:.2%} ({int(monthly.iloc[-1].correct)} / {int(monthly.iloc[-1].total)}). This is a material limitation. It is consistent with changing operating patterns or meter behavior, but the cause was not established. The full-year average must not be presented as stable performance throughout 2017. No tuning or model replacement was performed after this was observed.
"""
    report += """
## How to explain the results

Accuracy is the share of correctly recognized building-days. Macro F1 averages three class scores equally, so a common class cannot hide poor recognition of another. Validation scores chose the classifier; final test scores describe its later-date performance once selected. Model 1 and Model 3 answer different questions on different data, so their percentages are not an apples-to-apples competition.

The main limits are one campus, three meters, historical data, complete-day latency, and exclusion of poorly observed days. Results apply to the retained days. Equipment changes, new buildings, different meter calibration and future operational changes can reduce performance. No claim is made that this predicts physical circuit load types. Calendar activity classification would be a different task and is not presented here.

## Figures and traceable data

![Reserved test confusion matrix](test_confusion_matrix.png)
![Development model comparison](model_comparison.png)
![Development daily patterns](development_load_profiles.png)
![Test period stability](test_monthly_accuracy.png)
![Validation feature importance](validation_feature_importance.png)

All figure data is saved in results/*.csv and in chart_data.xlsx. Permutation importance uses only the final development validation fold and is descriptive; correlated features can share importance. Error bars on candidate comparisons show variability across three folds, not statistical significance. The Excel file includes editable charts, formulas for confusion counts and derived metrics, all test predictions, all CV scores, split counts, hourly patterns, monthly performance and feature definitions.
"""
    (O / "report.md").write_text(report, encoding="utf-8")
    print(f"Saved report and figures to {O}")

if __name__ == "__main__": main()
