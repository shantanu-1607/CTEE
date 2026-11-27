# Model 3: circuit-type and behaviour reassessment

Date: 27 November 2026.

## Correct objective

The requested first output is one of **R, L, C, RL, RC, RLC or LC**. The second output describes **constant voltage, constant current or constant impedance**, subject to confirmation of the professor's exact definition of constant voltage. These are two distinct targets, and a circuit's component combination does not automatically determine its regulation behaviour.

The current saved SVM recognizes **Academic, Facilities or Mess**. That was an incorrect objective for this project requirement. Its 85.07% building-recognition accuracy is not evidence of circuit-type or behaviour classification. Its existing model, workbook, predictions and results are retained as a separate historical experiment; their labels and scores have not been relabeled.

## What the supplied ZIP actually provides

All three meter CSVs have exactly these columns: `timestamp`, `power`, `current`, `voltage`, `frequency`, `power_factor`. They record whole-building electrical aggregates at approximately one-minute intervals. Neither circuit-type nor operating-behaviour labels appear in them. The calendar columns are `Date`, `working_day`, `activity`. The availability table records whether a building's data is present. The sole explanatory document identifies buildings and measurement conventions, not circuit schematics or verified class labels.

There are no supplied synchronized voltage/current waveforms, controlled voltage or frequency sweeps, switching-event circuit identities, signed reactive-power channel, verified leading/lagging convention, component values, circuit diagrams, regulation setpoints or behaviour experiment records. Existing measurements include negative power factors, but the supplied documentation does not explain their sign convention; treating a negative value as a verified capacitive label would be unjustified. Current also has missing entries, as recorded in the dataset audit.

## Why more features cannot establish the seven labels

For an ideal series RLC circuit under sinusoidal steady-state conditions,

`Z = R + j * (2*pi*f*L - 1/(2*pi*f*C))`.

At one frequency the measurement sees net reactance, not separate L and C values. Therefore a circuit containing both L and C can have the same terminal behaviour as one containing only L, only C, or neither reactive element. This remains a problem even if the full complex impedance is known; scalar building aggregates provide less information.

`feasibility/check_identifiability.py` demonstrates five pairs at 50 Hz: R versus RLC at resonance, RL versus RLC, RC versus RLC, L versus LC, and C versus LC. Within each pair, ideal terminal impedance, RMS current, real power and displacement power factor are equal. These are analytical counterexamples, **not new dataset labels, training samples or a claimed description of the actual buildings**. Their role is to show why a threshold on power factor cannot prove the physical component combination.

An RLC circuit's inductive and capacitive reactances can cancel, including at resonance. See [OpenStax: RLC series AC circuits](https://openstax.org/books/college-physics-ap-courses-2e/pages/23-12-rlc-series-ac-circuits). The matching examples are a direct calculation under the stated ideal-series-circuit assumptions. Real mixed building loads, harmonics and unknown topology add further ambiguity.

## Why constant voltage/current/impedance needs a defined experiment

A nearly steady voltage at a building meter can reflect a regulated supply. It does not prove that the load itself maintains constant voltage. Likewise, changes in current or power across a day can reflect equipment switching, occupancy and other changes rather than the response of an unchanged load to voltage.

Common power-system load models distinguish **constant impedance, constant current and constant power**. EPRI documents constant-impedance power as varying with voltage squared and constant-current power as varying linearly with voltage; the ZIP model combines these with constant power. See [EPRI OpenDSS load models](https://opendss.epri.com/PowerConversionElements.html). This does not authorize replacing the user's requested constant-voltage class with constant power.

If constant voltage is intentional, specify which voltage is regulated (for example, an electronic load's input terminal or a converter's output), what is varied, what range is tested, and the allowed regulation tolerance. If the professor means ZIP behaviour, confirm constant power explicitly. Different definitions require different measurements and labels.

## Replacement plan

1. Establish the target definitions, including whether component classes describe known physical topology or an equivalent fitted circuit, and whether series/parallel configurations are separate cases.
2. Obtain labeled measurements of known circuits or matching laboratory records for this dataset. For identification of physical component combinations, include informative transients and/or controlled frequency response in addition to synchronized voltage and current. Validate what combinations are distinguishable within the measured range.
3. Obtain behaviour labels from defined operating tests. Vary the appropriate independent quantity while keeping the circuit/device and other operating conditions controlled, and record the response and any controller mode.
4. Use separate outputs for circuit type and operating behaviour. Exclude circuit IDs, filenames and label-generating metadata from predictors. Keep all windows and sweeps from one physical circuit/device and experimental run together. Hold out independent circuits/devices or configurations for final evaluation, rather than randomly splitting nearby samples.
5. Compare appropriate classifiers on development groups only. Fit preprocessing inside each fold. Evaluate the locked final model on the untouched holdout, reporting per-class support, confusion matrices and unknown/ambiguous cases.

If real labeled data is unavailable, explicitly approved simulations can support a **simulation-based** experiment. Parameter sets and runs must remain grouped across splits. Such results must be labeled simulated and must not be reported as measured accuracy on this building dataset. Even simulations need an informative measurement design to avoid the exact ambiguities demonstrated above.

## Current decision

No circuit-type or behaviour model is claimed for the supplied ZIP. Do not create labels by thresholding power factor, voltage stability or V/I and then claim that a classifier's agreement with those same rules establishes real-world accuracy. The next training run depends on verified labels or an agreed replacement dataset/simulation and the clarified behaviour definition. The original ZIP, Model 1 and Model 2 are unchanged.
