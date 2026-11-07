# Electrical load classification

A supervised learning baseline for the supplied final dataset. One small neural
network predicts both load behaviour (CC, CP, CZ) and circuit type
(R, L, C, RL, RC, LC, RLC).

The model uses 19 distinct electrical measurements, two ReLU layers of 64 and
32 neurons, and separate 3-class and 7-class outputs. The network, gradients and
Adam updates are implemented with NumPy. Pandas and openpyxl read the workbook;
Matplotlib draws the result figures. There is no simulator, dashboard, physics
gate or ensemble in this baseline.

## Run

Use Python 3.12. From the repository root:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m load_classifier prepare
python -m load_classifier train
python -m load_classifier evaluate
python -m pytest -q
```

The saved baseline is included in `runs/baseline/`. Training refuses to overwrite
a populated run folder. To reproduce training, keep the same configuration and
split and choose another output folder:

```sh
python -m load_classifier train --run runs/reproduction
python -m load_classifier evaluate --run runs/reproduction
```

For consistent CPU timing, set `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1` and
`VECLIB_MAXIMUM_THREADS=1` before starting Python. Small floating-point differences
can still occur across machines. The training and split seeds are fixed at 42.

## Files

```text
configs/baseline.json       Network and training settings
data/                      Original workbook, fixed split and audit
load_classifier/data.py    Input checks, feature list and splitting
load_classifier/model.py   Shared MLP, gradients and Adam
load_classifier/train.py   Training and validation checkpoint selection
load_classifier/metrics.py Accuracy, F1 and confusion matrices
load_classifier/evaluate.py Reports, plots and prediction exports
runs/baseline/             Trained model and measured results
tests/                     Data, arithmetic and training checks
```

## Evaluation protocol

The workbook has 16,800 rows: 800 for each of 21 behaviour/circuit combinations.
Each pair has 400 normal and 400 tough examples. Allocation is 70/15/15 within
each pair and condition: 11,760 training, 2,520 validation and 2,520 test examples.
The saved `data/splits.csv` is reused across experiments.

Three columns are exact copies: V_L2 = Vrms, I_L2 = Irms, P_L2 = P. Dropping them
leaves 19 inputs. Sample IDs, labels and Source link metadata are never features.
Feature means and standard deviations are fitted on training rows only.

Both heads use cross-entropy, with equal task weights. Adam trains in batches of
128 with learning rate 0.001 and L2 coefficient 0.0001 on weights. Training runs
for at most 250 epochs, stopping after 25 epochs without sufficient validation
loss improvement. The best validation checkpoint is restored before testing.
No feature search or hyperparameter search is included in the baseline.

The main score is joint accuracy: both labels must be correct on the same row.
Per-head accuracy, macro-F1, per-class precision/recall/F1 and normal/tough results
are reported separately. See [the test report](runs/baseline/report.md) and
[exact metrics](runs/baseline/metrics.json).

The workbook documents simulated labelled loads based on real load-profile
values. CC/CP/CZ mean dominant ZIP behaviour. Unique IDs and source-reading keys
are checked, but no physical circuit IDs or generator are supplied. These scores
measure this dataset, not independently verified real-appliance performance.
See [data notes](data/README.md) for provenance.

## Predict a CSV

Provide the 19 measurement columns listed in `data/audit.json`. Extra columns
are ignored as model inputs; behaviour and circuit columns are optional.

```sh
python -m load_classifier predict --input measurements.csv --output predictions.csv
```

The export includes both predicted labels and probabilities for each class.
Probabilities are uncalibrated softmax outputs, not guarantees of correctness.

## Further experiments

Use validation scores to compare voltage-sweep ratios/slopes, feature groups,
network sizes and training settings. Put each run in a new folder and keep the
same split. Repeat promising settings with several seeds. Other classifiers can
then be compared under the same protocol. Do not choose settings using the test
scores or expect every change to improve accuracy.

The earlier simulation research and planning documents remain in Git history.
The current working tree contains only the supplied-dataset baseline.
