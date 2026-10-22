# From-scratch migration — completed NumPy pipeline

Requirement: NumPy is allowed; the learning algorithm must be inspectable and adjustable. The current executable path contains no prebuilt classifier, automatic differentiation, optimizer, signal-processing solver or library statistical test. Plotting, CSV and dashboard tools are auxiliary dependencies.

## Explicit algorithms

| Area | Implementation |
|---|---|
| Network | Shared arbitrary-width dense layers; 3/7 heads; manual forward pass and analytical backpropagation |
| Learned evidence | Head-specific sigmoid gates conditioned only on measured quality/excitation, with manually differentiated weight/bias gradients |
| Weight averaging | Optional explicit per-update EMA; raw/averaged checkpoint choice by validation loss |
| Objective | Stable softmax/CE, task masks/weights, class weights, smoothing, optional L2, measured-slope Huber consistency |
| Regularization | He/Xavier initialization; inverted dropout; train-only standardization and input clipping |
| Optimizers | Adam moments/bias correction/decoupled decay; SGD/momentum/Nesterov; global gradient clipping |
| Training | Seeded mini-batches; constant/cosine schedules; validation-loss early stopping and restored best weights |
| Spectra | Explicit radix-2 Cooley–Tukey butterflies; Hann window; direct harmonic phasor fitting |
| Fits | Reorthogonalized modified Gram–Schmidt QR; triangular back substitution; exhaustive small active-set nonnegative fit; centered slope regression |
| Dynamics | Explicit 2×2 implicit trapezoidal integration for RLC; first-order-hold recurrences for RL/RC; recursive coloured noise |
| Calibration | Grid-bracket/golden-section temperature minimization; confidence threshold search by probe budget; SNR gate |
| Evaluation | Confusion/F1, NLL/Brier/ECE, tie-aware rank AUC, Brier skill, continuous errors, group bootstrap/permutation |
| Statistics | Repeated-measures ANOVA with GG correction, Friedman tie correction, exact small-panel Wilcoxon and Holm adjustment; beta/gamma tails |
| Storage | NPZ arrays and JSON metadata with `allow_pickle=False`; no serialized Python estimator |

The distribution-tail representations are documented in [NIST DLMF §8.17](https://dlmf.nist.gov/8.17) and [§8.9](https://dlmf.nist.gov/8.9). The tests check independent closed forms rather than using SciPy as an oracle. FFT tests compare with the direct DFT equation. Neural derivative tests use centered finite differences for every trainable parameter, including masks, smoothing, priors, dropout and consistency terms.

## Verification and artifacts

The clean `.venv_numpy/` environment contains none of torch, scipy, sklearn or joblib. All 78 tests pass there. The static restriction test scans every core source file, not only the standalone network. Full results, configuration hashes, seeds and scope are in `RESULTS_NUMPY.md`. Runtime artifacts are isolated in `runs/numpy_v3/`.

The new `ctee/parameters.py` inspector reconstructs initialization, exports every learned value, and previews edits on an independent copy. Epoch histories record gradient norms and actual parameter movements; selected-run metadata records which raw/EMA weights were restored.

The checkpoint saves inference parameters/scaling, feature names, class order, architecture and training metadata. It does not preserve optimizer/RNG state for exact training resumption. The training budget is short enough to rerun experiments from the recorded seed/configuration.

## Scope changes from checkpoint 1 and the initial plan

The old PyTorch/scikit-learn training, SciPy solver/NNLS/filter and library metrics are removed from the current executable path. The old random-forest comparator is retired; comparisons use manually trained baselines, a feature ablation, physics-enhanced models and analytical electrical rules. Separate single-task models, five-seed studies, AVME/Hilbert reproduction, parameter-search automation and independent ngspice/hardware validation are not implemented and are not claimed as completed.

The normal workflow does not load the legacy framework checkpoints or their published metrics. Historical files remain available for provenance. Current scores are new measurements of the NumPy pipeline. Some numerical routines intentionally target this small problem (three-component fits, exact signed-rank enumeration for at most twenty blocks); they are not general-purpose replacements for mature numerical libraries.

Scientific novelty is supported only to the extent shown by the ablations and acquisition experiments. A manual neural network alone is not novel; a globally best classifier and a particular grade are not established.
