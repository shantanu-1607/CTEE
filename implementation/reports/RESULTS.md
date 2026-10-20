# Measured results - research prototype

Selected integrated model: **physics_enhanced_11**, chosen on validation data before final testing.

**Scope:** synthetic controlled-source emulator; no claim of appliance/hardware accuracy or global best classifier.

| Model | Clean joint accuracy | 20 dB joint accuracy | Sensor shift | Frequency shift | Coloured 10 dB |
|---|---:|---:|---:|---:|---:|
| baseline_clean_11 | 56.67% | 9.63% | 7.04% | 9.63% | 12.22% |
| baseline_robust_11 | 94.44% | 88.15% | 51.85% | 81.11% | 46.67% |
| features_only_11 | 100.00% | 93.33% | 67.41% | 94.81% | 62.96% |
| physics_enhanced_11 | 100.00% | 96.30% | 74.81% | 97.04% | 72.96% |
| physics_enhanced_22 | 100.00% | 94.81% | 64.81% | 92.96% | 71.11% |
| physics_enhanced_33 | 100.00% | 95.93% | 74.07% | 93.70% | 68.89% |
| rf_baseline | 80.74% | 73.33% | 69.26% | 67.41% | 54.81% |
| rf_enhanced | 99.63% | 93.70% | 67.04% | 92.22% | 71.11% |
| electrical_rules | 100.00% | 72.59% | 42.59% | 75.93% | 44.81% |

20 dB gain over noise-trained baseline: 9.05%; physical-group bootstrap 95% CI [0.04678571428571429, 0.1349206349206349].

## Acquisition at 20 dB

| Policy | Mean probes | Coverage | Accepted joint accuracy |
|---|---:|---:|---:|
| fixed | 4.23 | 76.67% | 97.58% |
| random | 4.17 | 76.67% | 98.55% |
| adaptive | 3.47 | 76.67% | 97.10% |

Feature + model compute p95: 0.84 ms on the local CPU, excluding acquisition and UI.

## Interpretation

Use the noise-trained baseline for the fair main comparison. The clean-only baseline is a distribution-shift demonstration, not the sole baseline. An added physics penalty is supported only if its ablation improves the measured result. The analytical rules or random forest may match or beat a neural variant; report that honestly.

Three physics-model training seeds are included. Other variants currently use one seed. Test confidence intervals resample physical branches, not windows. Ten simulator device panels support a supplemental ANOVA/Friedman analysis; they are not real laboratories.

The initial coloured-noise stress test exposed confident errors. A measured-SNR operating-range guard was subsequently calibrated on calibration devices only; the original evaluation is archived in reports/history. A fresh independent configuration pool checks the frozen classifier and guard in fresh_confirmation.json. Low coverage/rejection is not increased classification accuracy.

Partial-acquisition calibration, independent ngspice verification, broader controller families, real datasets with verified dual labels, hardware, and external generalisation remain open validation work. See all JSON/NPZ exports for exact values and failure cases.
