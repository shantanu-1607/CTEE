# Later implementation: ordered checklist

**Planning document only. No items below have been implemented by preparing this folder.**

## Gate 0 - agree what the labels mean

- [ ] Take `03_PROFESSOR_DISCUSSION.md` to the professor.
- [ ] Record the measurement port and the exact meaning of composition.
- [ ] Agree series topology, treatment of active controllers, supported class pairs, and simulation-only scope.
- [ ] Confirm deadline, team size, deliverable format, and whether hardware is compulsory.
- [ ] Freeze a short label contract and include actual circuit/controller diagrams later.

**Pass condition:** the data can be labelled without assigning physically false CC/CP/RLC combinations.

## Gate 1 - build and audit the pilot dataset

- [ ] Choose the simulator and pin the environment.
- [ ] Implement passive circuit families and explicitly model regulated fixtures.
- [ ] Check analytical impedance, transient settling, energy balance, and numerical convergence.
- [ ] Validate terminal CC/CP/CZ behaviour across the declared voltage range.
- [ ] Create a supported-pair matrix; retain ambiguous cases with explicit status.
- [ ] Collect pilot waveforms with provenance and quality flags.
- [ ] Spot-check passive cases in an independent circuit simulator.
- [ ] Set final parameter ranges and estimate storage/runtime from the pilot.

**Pass condition:** every labelled fixture has a defensible physical explanation and the sampling captures the required phenomena.

## Gate 2 - freeze data handling

- [ ] Split by parent physical configuration: 60/15/10/15 train/validation/calibration/test.
- [ ] Keep all modes, probes, crops, and noisy descendants together.
- [ ] Audit duplicate/near-duplicate configurations across splits.
- [ ] Fit imputation and scaling using training data only.
- [ ] Save class order, feature dictionary, units, masks, and split manifests.
- [ ] Overlap nominal power ranges across classes to reduce shortcuts.

**Pass condition:** a held-out device cannot be reconstructed from a training duplicate or noisy copy.

## Gate 3 - complete the required baseline

- [ ] Implement electrical rules, random forest, and single-task MLP comparators.
- [ ] Train the shared 96 -> 64 embedding with 3-class and 7-class heads.
- [ ] Check loss masking when a head lacks a valid ground-truth label.
- [ ] Save and reload the complete preprocessing/model bundle.
- [ ] Produce per-head confusion matrices, macro-F1, and joint exact-match accuracy.
- [ ] Demonstrate one valid prediction from an entirely saved/reloaded model.

**Pass condition:** the 4-mark development/deployment requirement has a working milestone.

## Gate 4 - implement and isolate enhancements

- [ ] Add voltage-response, transient, and frequency fingerprints.
- [ ] Add realistic waveform corruption and recompute features.
- [ ] Add masked physics penalties; include a zero-weight control.
- [ ] Fit temperature scaling and acceptance thresholds on calibration data.
- [ ] Implement a deterministic adaptive measurement policy with a maximum budget.
- [ ] Check that unchosen probe responses and truth never enter policy decisions.
- [ ] Generate electrical evidence explanations and grouped feature importance.

**Pass condition:** each claimed contribution has a runnable variant and a corresponding ablation.

## Gate 5 - test honestly

- [ ] Freeze the experiment manifest before opening final test results.
- [ ] Run the baseline/variant table, preferably with five neural training seeds.
- [ ] Run clean, 20 dB, parameter-shift, and ambiguity tests as the minimum condition set.
- [ ] Compare adaptive/fixed/random policies at the same probe budget.
- [ ] Report acquisition time separately from feature/model/UI compute time.
- [ ] Plot reliability and risk-versus-coverage curves, including low-coverage failures.
- [ ] Use group-level confidence intervals and paired comparisons.
- [ ] Conduct the planned panel-based ANOVA only with suitable data/assumptions; otherwise justify an alternative.
- [ ] Report continuous errors only for continuous quantities and never for integer class codes.
- [ ] Show at least three failure cases and avoid selecting only favourable examples.

**Pass condition:** improvement claims survive the relevant controls; unsupported claims are removed or rewritten.

## Gate 6 - package and rehearse

- [ ] Complete local dashboard replay and CSV validation.
- [ ] Include class definitions, both heads, electrical plots, confidence, and next-measurement status.
- [ ] Reproduce results from the saved environment/configurations.
- [ ] Prepare report, comparison table, ablation figure, robustness figure, calibration figure, and latency table.
- [ ] Prepare the four-minute demonstration and rehearse viva questions.
- [ ] Freeze the submission before the final session; reserve time to recover from setup issues.

**Scope rule:** cut hardware, heavy AVME reproduction, large networks, and online adaptation before cutting baseline deployment or evidence quality.
