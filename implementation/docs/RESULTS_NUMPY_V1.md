# NumPy experiment results — frozen run v1

Measured 22 October 2026. These scores belong to the manual NumPy implementation, not the earlier framework prototype. Both heads must be correct for **joint accuracy**.

## Dataset and protocol

- 700 physical configurations: 100 per composition. Split by parent circuit: 420 train, 105 validation, 70 calibration, 105 test. Modes, noisy copies, and partial-probe records stay with their parent circuit.
- 31,050 feature records are derived observations, not 31,050 independent devices. Each test condition contains 270 supported behaviour/composition episodes.
- 18 supported pairs under the series-branch controlled-source contract; CP with pure L/C/LC is excluded. This is an ideal simulator, not arbitrary appliance identification.
- Frozen experiment settings: `configs/experiments.json`, data seed 20261022. Three training seeds each for the robust baseline and enhanced model: 11, 22, 33.
- Checkpoints selected by validation loss. The deployed variant was selected on validation joint accuracy and composition macro-F1, before final testing. Temperatures, per-budget acceptance thresholds and SNR guard use calibration devices only.
- Selected model: **physics_enhanced_11**, 16,074 trainable parameters, shared 95→96→64 encoder and separate 3/7-class heads. Baselines have 70 ordinary features; all use the same hidden widths and noise-training budget.

## Held-out classification results

| Model | Clean | White noise 20 dB | Sensor shift | Frequency shift | Coloured noise 10 dB |
|---|---:|---:|---:|---:|---:|
| baseline_robust_11 | 98.89% | 93.33% | 59.63% | 88.15% | 46.30% |
| features_only_11 | 100.00% | 96.67% | 63.33% | 94.44% | 44.44% |
| physics_enhanced_11 | 100.00% | 96.67% | 72.96% | 94.07% | 55.93% |
| electrical_rules | 100.00% | 77.04% | 39.26% | 71.85% | 37.78% |

Sensor shift includes voltage/current gain error and one-sample channel misalignment, alongside 20 dB noise. Frequency shift changes all probe source frequencies by 20%, alongside 20 dB noise. The severe coloured-noise condition is outside the training noise range.

Across all three training seeds (mean ± sample standard deviation in percentage points):

| Family | Clean | White noise 20 dB | Sensor shift | Frequency shift |
|---|---:|---:|---:|---:|
| baseline_robust | 99.01 ± 0.21 | 94.57 ± 1.13 | 59.38 ± 10.74 | 88.52 ± 1.34 |
| physics_enhanced | 100.00 ± 0.00 | 96.91 ± 1.13 | 70.00 ± 3.92 | 94.32 ± 0.77 |

The selected model's 20 dB behaviour macro-F1 is 1.0000 and composition macro-F1 is 0.9682. The nine joint errors are composition errors: RL→RLC (2), LC→L (2), RLC→RL (5). Features-only and the selected physics variant tie at 96.67% joint accuracy at 20 dB. The combined prior/penalty improves some stress conditions, but this experiment does not isolate a penalty-only benefit.

The clean-only ablation underfit (33.33% clean joint accuracy). Its best epoch was 1 under noisy validation-loss selection. Do not use this unsuccessful run as the main evidence of improvement.

## Statistical evidence and its limits

The selected model improves episode-pooled 20 dB accuracy by **3.33 percentage points** over baseline seed 11. Averaging each physical branch's supported modes first gives **4.29 points**, bootstrap 95% CI **[0.95, 7.94] points**, paired group sign-permutation p=0.0234 over 105 branches.

A supplemental ten-panel repeated-measures ANOVA gives GG-corrected p=1.24e-8; Friedman p=3.99e-5 across four models including electrical rules. However, the baseline–selected-model panel Wilcoxon comparison is **not significant after Holm correction (p=0.625)**. These tests use different units and hypotheses. Do not claim that the omnibus ANOVA proves the enhanced neural model beats every comparator.

## Confidence and measurement cost

| Policy, 20 dB | Mean probes | Coverage | Joint accuracy among accepted |
|---|---:|---:|---:|
| fixed | 4.219 | 78.15% | 97.63% |
| random | 4.122 | 78.15% | 97.16% |
| adaptive | 3.437 | 78.15% | 98.10% |

Adaptive acquisition reduces mean probes by **18.5%** versus the fixed order in matched episodes. Its budget-three joint accuracy is 96.67%, compared with fixed-order 85.56%. The policy is deterministic and evidence-guided; no reinforcement learner or information-gain optimality is claimed. Operational inference stops at acceptance; only offline diagnostic traces continue to draw equal-budget curves.

All severe coloured-noise recordings are rejected by the calibrated quality guard (zero coverage). The raw classifier still gets only 55.93% correct on that test condition. Rejection is a useful reliability mechanism, not an accuracy of 100%.

At 20 dB, composition NLL changes from 0.11677 to 0.09870 after calibration, Brier score from 0.04729 to 0.04684, and ten-bin ECE from 0.01580 to 0.01300. Temperature scaling does not change the argmax labels. AUC and Brier skill scores for both heads are in the JSON report. Regression metrics apply to measured voltage-response slope, never to integer class codes; MAPE is omitted because the CC slope is zero.

## Fresh independent confirmation

A frozen model and thresholds were also tested on **140 new configurations**, 360 supported episodes per condition, seed 20261023. No settings were tuned on this pool.

| Condition | Enhanced joint | Baseline joint | Coverage | Accepted joint |
|---|---:|---:|---:|---:|
| clean | 100.00% | 98.89% | 100.00% | 100.00% |
| noise20 | 99.17% | 97.50% | 84.17% | 99.34% |
| coloured10 | 60.28% | 53.61% | 0.00% | undefined |

The fresh-pool improvement is encouraging but its device-level confidence interval includes zero. This remains validation within one simulator family.

## Reproducibility and implementation checks

- 64 tests passed in a clean Python 3.12 environment. PyTorch, SciPy, scikit-learn and joblib are not installed in that environment.
- Tests include all-parameter finite-difference gradients, independent direct DFT comparisons, optimizer updates by hand, Kirchhoff residuals, time-step convergence, group separation, checkpoint equivalence, CSV roundtrips, causal probe access, abstention, and dashboard classification.
- Physics audit passes all 18 supported pairs. Maximum measured voltage-slope error is 1.79e-7; regulation relative error is 7.61e-8. These checks do not replace an independent simulator/hardware test.
- Local CPU feature-plus-model p95: 1.03 ms, 100 repetitions; excludes recording, settling, actuation, UI and network. This is a local benchmark, not a hardware deadline guarantee.
- Dataset SHA-256: `8a41e5c9f50efc9e8c5457c87fde169e0cea6baf710a5f8f84696d7a02adcb34`.
- Experiment configuration SHA-256: `e8f25ebc055ee06bf9f934aedc7eb5d9e7da1983bb3954e00837c399bf79596c`.
- Selected checkpoint SHA-256: `28a60c15d0d44159e42743a56b33240f2c78c78006707e20c649387bad0e8ea0`.

Generated arrays, checkpoints, raw predictions, calibration, plots and complete reports live in ignored `runs/numpy_v1/`. Reproduce them with `Run_All_Phases.command`. Historical framework results under the old root `reports/` are retained as history and are not loaded by the current dashboard.

## Defensible contribution

The proposed project contribution is the evaluated combination of voltage-response/spectral/startup fingerprints, a shared two-head model with electrical priors and a masked consistency loss, calibrated abstention, and causal adaptive measurement. Implementing these algorithms manually makes their parameters inspectable; manual implementation alone is not scientific novelty.

Real hardware, independent ngspice validation, converter dynamics, arbitrary topologies and real-appliance dual labels remain outside the validated scope. Supplied-paper appliance scores cannot be compared numerically with this different benchmark. No worldwide novelty, global-best classifier, or guaranteed grade is claimed.
