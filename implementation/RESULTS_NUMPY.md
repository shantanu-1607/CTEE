# Current NumPy results — learned weights and physics gates

Measured 22 October 2026. Current run: **numpy_v3**; deployed checkpoint **physics_gated_33**. The old report is preserved in `docs/RESULTS_NUMPY_V1.md`. Do not compare percentages across different circuit pools as if they were paired measurements.

## The actual ML parameters

The model has **25,668 trainable weights/biases**, including 26 values in two quality-conditioned sigmoid gates. Training changed **25,666** values by more than 1e-12. The two unchanged gate entries correspond to the always-present nominal-probe indicator, which becomes zero after standardization; this is an expected constant-input effect.

All parameters are explicit NumPy arrays. Manual backpropagation, Adam updates and EMA modify actual model weights. The selected checkpoint uses **ema weights from epoch 50**, after a run with 9,144 optimizer steps and restoration of the best validation checkpoint. The dashboard's **Weights and learning** tab exposes arrays, changes from initialization, gradient/update traces, a complete CSV download and isolated single-weight edits.

## Protocol

- 700 new physical circuits: 420 train, 105 validation, 70 calibration, 105 test. Data seed 20261024. The fresh split was not used in the earlier v1 experiment.
- 31,050 original derived records. Domain randomization adds 16,200 training records and 4,050 validation records; these remain copies of their original physical groups, not new independent circuits.
- Randomization uses shared per-episode voltage/current gain errors, fractional channel delay, noise from 12–35 dB and AR(1) coefficient 0–0.85. All neural families receive the same records, including the robust baseline.
- Four families, three candidate hyperparameter settings each. Candidate selection uses condition-balanced validation joint accuracy, with composition macro-F1 and NLL for ties. The chosen setting for each family is run with seeds 11/22/33. Early stopping/checkpoint choice uses validation loss.
- A validation-only pilot without EMA is archived under `runs/numpy_v2/`. The matched follow-up uses EMA decay 0.995 for every family, selecting raw or averaged weights by validation loss. Test results were not used to make this change.
- Classifier, gates, features and calibration were frozen before testing. A separate pool of **700 additional physical circuits**, seed 20261025, provides 1,800 episodes per condition for confirmation.
- This remains the declared ideal controlled-source series-branch family with 18 supported label pairs. It is not a real-appliance or hardware benchmark.

## Held-out joint accuracy

Both labels must be correct. There are 270 supported episodes per test condition.

| Model | Clean | 20 dB white | Sensor shift | Frequency shift | 10 dB coloured |
|---|---:|---:|---:|---:|---:|
| baseline_robust_11 | 98.15% | 97.78% | 96.67% | 97.04% | 78.15% |
| features_only_11 | 100.00% | 99.63% | 97.04% | 97.78% | 77.78% |
| physics_enhanced_11 | 99.26% | 98.52% | 94.44% | 98.89% | 81.48% |
| physics_gated_11 | 99.26% | 99.63% | 96.67% | 98.15% | 87.41% |
| physics_gated_33 | 100.00% | 99.26% | 94.81% | 98.15% | 80.00% |
| electrical_rules | 100.00% | 80.37% | 46.67% | 79.26% | 44.44% |

The deployed model was chosen by validation; it is not the best test seed on every condition. Its two 20 dB errors are RL→RLC and RLC→RL. Behaviour macro-F1 is 1.0000 and composition macro-F1 is 0.99365.

Across three training seeds (mean ± sample standard deviation in percentage points):

| Family | Clean | 20 dB white | Sensor shift | Frequency shift | 10 dB coloured |
|---|---:|---:|---:|---:|---:|
| baseline_robust | 98.52 ± 0.37 | 97.78 ± 0.37 | 96.42 ± 0.77 | 96.67 ± 0.98 | 76.30 ± 3.53 |
| features_only | 99.63 ± 0.64 | 98.77 ± 0.93 | 95.80 ± 1.13 | 98.02 ± 0.43 | 79.14 ± 1.30 |
| physics_enhanced | 99.63 ± 0.37 | 98.64 ± 0.21 | 94.07 ± 0.37 | 98.40 ± 0.57 | 80.62 ± 2.88 |
| physics_gated | 99.75 ± 0.43 | 99.38 ± 0.21 | 95.56 ± 0.98 | 98.02 ± 0.21 | 82.96 ± 3.92 |

The gated family has the highest three-seed mean for 20 dB and coloured noise. The ordinary baseline is stronger on the sensor-shift test. This is evidence of a tradeoff, not universal dominance.

## Do the learned gates do anything?

Replace only the gate logits with zeros, forcing gate=0.5; retain exactly the same trained neural weights. This frozen-weight counterfactual is different from training a fixed-prior model from scratch.

| Condition | Learned gates | Same weights, gate=0.5 |
|---|---:|---:|
| clean | 100.00% | 99.26% |
| coloured10 | 80.00% | 75.56% |
| frequency_shift | 98.15% | 97.04% |
| noise20 | 99.26% | 98.15% |
| noise30 | 100.00% | 99.63% |
| sensor_shift | 94.81% | 95.56% |

Mean composition-gate value decreases from **0.850 on clean recordings to 0.547 under severe coloured noise**. Thus the fitted model adjusts its trust in physical evidence using measured quality. These gate values are not calibrated probabilities that physics is correct. The counterfactual improves several conditions but hurts sensor-shift accuracy; it does not isolate conditional gating from every possible global learned prior scale.

## Fresh independent confirmation

The model was not changed after the held-out test. Each condition below has 1,800 episodes from 700 new physical configurations.

| Condition | Selected model | Baseline seed 11 | Coverage | Accepted joint accuracy |
|---|---:|---:|---:|---:|
| clean | 100.00% | 99.06% | 100.00% | 100.00% |
| noise20 | 97.89% | 97.78% | 83.56% | 98.87% |
| coloured10 | 79.00% | 76.83% | 0.00% | undefined |

The large confirmation pool gives a more conservative 20 dB result than the smaller test set. Its 20 dB improvement is **not statistically established**: physical-device-average gain 0.238 percentage points, 95% bootstrap CI [−0.619, 1.071], sign-permutation p=0.597. Clean and coloured-noise comparisons are favourable, but those individual p-values are exploratory, unadjusted for these multiple conditions.

On the original test split, the selected-model 20 dB device-average gain is 1.905 points, CI [−0.317, 4.603], also including zero. The omnibus GG-ANOVA/Friedman tests across neural variants and electrical rules are significant; neural pairwise panel tests are not significant after Holm adjustment. Do not use the omnibus result to claim the novel model beats all alternatives.

## Acquisition and confidence limitations

**Full measurement is the dashboard default**, using all five probes and the headline classifier. Adaptive mode is available as an explicit accuracy/cost tradeoff:

| Policy at 20 dB | Mean probes | Coverage | Accepted joint accuracy |
|---|---:|---:|---:|
| fixed | 4.193 | 80.74% | 98.62% |
| random | 4.144 | 80.74% | 98.62% |
| adaptive | 3.385 | 80.74% | 96.33% |

Adaptive saves about 19.3% of probes versus the confidence-stopped fixed order, but its accepted accuracy is lower. This run does **not** demonstrate equal-accuracy acquisition savings. Offline diagnostic traces continue after the actual stop only to compare observation budgets; operational inference stops without reading future probes.

The SNR gate rejects all 10 dB coloured-noise cases. Their raw classifier accuracy remains 80% on the main test and 79% on confirmation. Zero coverage is rejection, not 100% accuracy.

Temperature scaling improved its calibration objective but **worsened test composition NLL** from 0.02487 to 0.09012 and Brier from 0.01432 to 0.01501. A confidence-calibration benefit is not established for this run. This does not change argmax classification accuracy. Report the before/after confidence metrics rather than assuming scaling always helps.

## Verification and reproduction

- **78 tests passed** in a clean Python 3.12 environment without PyTorch, SciPy, scikit-learn or joblib.
- Tests verify every parameter's analytical gradient, including learned gates; actual parameter updates and EMA by hand; shared-task gradient addition; checkpoint equivalence; source restrictions; no test leakage in search; causal acquisition; CSV; circuit equations; and interactive parameter edits.
- Feature-plus-model local CPU p95: **1.03 ms**, excluding recording, settling, actuation and UI.
- The electrical audit covers all 18 supported pairs. Independent ngspice/hardware validation is not performed.
- Dataset SHA-256: `f6bd4f661c1dfd6f2564c82e2b38a6c116ef2ea23202ecbd1098761089b4e256`.
- Configuration SHA-256: `19054159f6a15e0e946f4e7731cc2ec28636cfd281b3e14b1ecba59ed90d8e34`.
- Selected checkpoint SHA-256: `7bdfee795245ed1a78b62edea4a6314f22b8f9fbb84fd759cd8bfc9cef0671f7`.

Source/configuration and the compact JSON result record are committed. Full training histories, augmentation provenance, weights, raw predictions, plots and reports are in ignored `runs/numpy_v3/`. Run `Run_All_Phases.command` to reproduce with the pinned dependencies. The v1/v2 runs remain separate history.

## Contribution statement

The proposed contribution is measured-quality-conditioned learned electrical priors for this shared dual-target task, with manually differentiated gates, direct learned-weight inspection, robust training and causal measurement control. EMA/gating/calibration are established components and are credited in `docs/WEIGHTS_AND_NOVELTY.md`. The implementation is useful and the classifier is strong on its benchmark; worldwide uniqueness, global optimality, universal improvement and a particular grade remain unproven.
