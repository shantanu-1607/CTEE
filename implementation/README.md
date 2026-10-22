# CTEE LoadLab — manual NumPy implementation

One shared neural encoder predicts **CC/CP/CZ behaviour** and **R/L/C/RL/RC/LC/RLC branch composition**. The network, derivatives, optimizers, Fourier analysis, circuit fitting, transient solver, calibration and metrics are implemented explicitly. NumPy supplies arrays and arithmetic. Streamlit, pandas and plotting packages handle the interface and files.

**Current result:** the validation-selected learned-gate model has 25,668 trainable weights/biases and achieves 100% clean and 99.26% at 20 dB joint accuracy on held-out simulated circuits. The matched baseline reaches 97.78% at 20 dB. A larger independent pool of 700 new circuits gives 97.89%; its noisy improvement over the baseline is statistically inconclusive. See [RESULTS_NUMPY.md](RESULTS_NUMPY.md) for all seeds, failures, coverage, statistics and limitations. These are simulator results, not appliance/hardware accuracy or proof of a globally best model.

## Run the dashboard

On this computer, double-click `Run_Dashboard.command`, or:

```sh
cd "/Users/shantanusingh/Desktop/CTEE project/implementation"
.venv_numpy/bin/python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

Open <http://127.0.0.1:8501>. Choose a new simulated fixture, a held-out recording, or a measured CSV. Both predictions appear with confidence, electrical evidence, learned physics trust, waveforms and an accepted/inconclusive status. Full measurement is the default for classification; the adaptive option trades some accuracy for fewer probes. The **Weights and learning** tab displays every actual learned weight/bias, initialization changes and gradient/update traces, and previews isolated single-value edits. The adaptive policy stops when its evidence and confidence gates pass. Physical active probing is not implemented.

## Fresh installation

Use Python 3.12 and `uv`, or Python's venv and pip:

```sh
uv venv .venv_numpy --python 3.12
uv pip sync requirements.lock --python .venv_numpy/bin/python
```

Alternative:

```sh
python3.12 -m venv .venv_numpy
.venv_numpy/bin/python -m pip install -r requirements.lock
```

No torch, TensorFlow, scikit-learn, SciPy, joblib or automatic differentiation is required. The legacy `.venv/` may still contain earlier prototype dependencies; the current launchers use the clean `.venv_numpy/` environment.

## Reproduce phase by phase

Run from this `implementation/` directory:

```sh
.venv_numpy/bin/python -m ctee audit
.venv_numpy/bin/python -m ctee generate
.venv_numpy/bin/python -m ctee train
.venv_numpy/bin/python -m ctee evaluate
.venv_numpy/bin/python -m ctee confirm
.venv_numpy/bin/python -m pytest -q
```

`Run_All_Phases.command` runs these phases and tests together. Generated files go to ignored `runs/numpy_v3/`; it may overwrite that run. Keep a new experiment separate:

```sh
.venv_numpy/bin/python -m ctee all --root runs/my_experiment --config configs/gated_experiments.json
```

The app always displays the standard `runs/numpy_v3/` run. A new clone contains code/configuration and the checked-in result summary; generate/train before opening its dashboard. Seeds and arrays are reproducible within the pinned environment, though timings vary and floating-point details may vary across platforms.

## Completed phases

| Phase | Implemented | Evidence |
|---|---|---|
| 1 — Electrical contract | 18 valid pairs; controlled-source emulator; steady-state and startup checks | `docs/LABEL_CONTRACT.md`, physics audit, KVL/convergence tests |
| 2 — Data/features | 700 parent circuits; disjoint four-way split; 95 features; raw examples and seeds | `ctee/data.py`, `runs/numpy_v3/reports/data_audit.json` |
| 3 — Baselines | Shared 70-feature two-head MLP; three robust training seeds; analytical rules | `configs/gated_experiments.json`, validation/test exports |
| 4 — Enhancements | Response/spectral/startup features; trainable quality gates; masked consistency; EMA of actual weights; quality guard | Four families, equal candidate budgets and three seeds each |
| 5 — Adaptive measurement | Nominal → voltage → harmonic/frequency probes; causal stopping | Matched fixed/random/adaptive budget curves; no-oracle tests |
| 6 — Deployment/evaluation | Local dashboard; CSV input/export; fresh confirmation; manual statistics | 78 tests and `RESULTS_NUMPY.md` |

## Change the parameters

Edit `configs/gated_experiments.json` before running a new experiment. Exposed settings include hidden widths, activation, dropout, initialization, precision, learning rate, batch size, epochs, patience, optimizer, decay, Adam betas/epsilon, gradient clipping, learning-rate schedule and label smoothing. Each variant declares extra features, prior strength, physics-loss weight and learned gating. `training_seeds`, `candidates`, `augmentation` and `ema_decay` make the search and actual parameter averaging reproducible. Gates add trainable weights/biases; EMA averages the learned parameter arrays, not predictions.

`ctee/scratch/network.py` and `ctee/scratch/training.py` expose further controls through `NetworkConfig` and `TrainingConfig`, including scale/clipping controls, class weights, Huber delta, SGD momentum and Nesterov. Backpropagation and optimizer updates are visible in plain Python. Do not alter the test split repeatedly to select better scores. Use validation for tuning and reserve a genuinely new confirmation pool after further development.

## Where the work lives

| File | Purpose |
|---|---|
| `ctee/simulator.py` | Waveforms, regulation, noise and explicit transient integration |
| `ctee/numerics.py`, `ctee/features.py` | QR fitting, radix-2 FFT and named electrical features |
| `ctee/scratch/` | Neural equations, shared gradients, optimizers and training loop |
| `ctee/train.py`, `ctee/model.py`, `ctee/research.py` | Electrical-data adapter, matched validation search, selection, calibration and checkpoints |
| `ctee/parameters.py`, `ctee/gate_analysis.py` | Inspect and preview parameter edits; controlled frozen-gate ablations |
| `ctee/augmentation.py` | Development-only waveform sensor/noise randomization |
| `ctee/inference.py`, `ctee/quality.py` | Confidence/evidence gates and acquisition decisions |
| `ctee/metrics.py`, `ctee/statistics.py` | Classification/probability metrics, paired inference and ANOVA |
| `app.py` | Local dashboard; it loads only new NumPy artifacts |
| `docs/VIVA_GUIDE.md` | Simple explanation and defensible answers for the presentation |

Read `docs/WEIGHTS_AND_NOVELTY.md` for the exact learnable-gate equations and prior-work credit. EMA and gating are established components; the proposed contribution is their measured-quality-conditioned electrical application and controlled evaluation. Confidence calibration worsened test scores in this run, and adaptive acquisition did not preserve full accuracy; neither is advertised as a universal improvement.

The root planning documents record the original proposal. The implementation contract and current results supersede their candidate ranges, framework choices, sample lengths and unimplemented stretch experiments. Historical root `artifacts/`, `data/` and `reports/` belong to the earlier prototype and are not loaded by the current app.

To reproduce the historical v1 protocol explicitly, use `--root runs/numpy_v1 --config configs/experiments.json`. The v2 directory records the validation-only pilot; the current default is v3.
