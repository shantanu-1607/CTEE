# CTEE project: implementation and research plan

**Proposed title:** Uncertainty-Guided, Physics-Consistent Multi-Task Electrical Load Classification

**Prepared:** 20 October 2026 | **Status:** proposal and implementation plan only

**Assignment:** 4 marks for model development/deployment; 11 marks for implemented and validated enhancements. All numerical goals below are targets, never measured results. The enhancement sub-allocation suggested below is our evidence plan, not an official marking scheme.

## 1. Recommended project and success criteria

Build one integrated neural network with a shared feature extractor and exactly two classification heads: a 3-class CC/CP/CZ head and a 7-class R/L/C/RL/RC/LC/RLC head. A local dashboard will display both outputs, electrical evidence, confidence, and the baseline comparison.

The central research question is: **Can electrical consistency and uncertainty-guided measurement improve both-label reliability under noise and ambiguous operating conditions, while using fewer measurements than a fixed acquisition schedule?**

The baseline is a shared multilayer perceptron using ordinary electrical features. The enhanced system adds voltage-response and dynamic fingerprints, valid physics penalties, realistic noise training, confidence calibration, and a small adaptive acquisition policy. Start with this compact model; a large transformer or elaborate reinforcement-learning agent is unnecessary.

| Requirement | Concrete evidence to produce later |
|---|---|
| Integrated model | One saved model; one shared embedding; 3-way and 7-way outputs in the same forward pass |
| Correct testing | Unseen physical configurations, held-out noise conditions, both-head confusion matrices, joint accuracy |
| Deployment | Reloadable local application using the saved preprocessing and weights |
| Implemented enhancement | Baseline/variant table, ablation study, matched-measurement comparison |
| Credible novelty | Specific hypothesis, attribution to prior work, honest limitations and failure cases |

**Must-have:** complete baseline, physically valid labels, two-head deployment, advanced features, robustness study, calibration, and controlled comparisons. **Main showcase:** adaptive measurement in simulation. **Stretch only:** hardware, exact AVME reproduction, self-supervised pretraining, online adaptation, ONNX export.

## 2. Resolve the electrical definitions before generating data

Behaviour and composition are different targets. At fixed frequency, a fixed passive RLC network normally has constant impedance after settling. It does not become a constant-power load by changing its label. Regulated CC/CP behaviour requires an active controller or another explicitly described mechanism. EPRI's load-model documentation distinguishes the three voltage laws [W1].

**Recommended experimental contract:** circuit composition means the documented passive input network in a controlled load fixture. Draw and record that network separately from the controller. The behaviour label describes the measured terminal response in the tested voltage range. Use a fixed series connection convention for the seven network classes in the main benchmark. The controller is present only in fixtures designed for regulated modes. This is a scoped interpretation to discuss with the professor, not a hidden reinterpretation of the assignment.

Maintain three data subsets: **passive**, with exact series RLC topology and CZ behaviour; **regulated**, with verified CC/CP behaviour and composition only when its input network is known; and **paired**, where both labels have valid ground truth. Every experiment reports which subset and label support it uses. Missing composition labels are masked during training; they are never guessed from an appliance name.

Do not require all 3 x 7 pairs. Verify supported pairs through simulation and an operating-point audit. A pure lossless L/C network has essentially zero active power, so assigning it a nonzero constant-active-power label is invalid. An L/C input network in front of an active power-consuming converter is a different, explicitly drawn composite device. CC/CP claims must hold at the measurement port, including network losses and controller limits.

**Decision gate:** obtain agreement on the measurement port, series topology convention, treatment of controlled fixtures, and whether simulation-only deployment is acceptable. If the professor requires literal passive-only composition of the complete device, explain the incompatibility with independent CC/CP labels and agree a restricted supported-pair benchmark. Do not silently change definitions.

Even perfect measurements at one sinusoidal frequency cannot uniquely reveal arbitrary internal topology. Different circuits can share the same terminal impedance. Limit claims to the specified family, parameter range, excitation, and bandwidth. Ambiguous examples remain an explicit test category.

## 3. What the supplied papers establish

The papers are related research sources, not additional instructions. Their tasks and labels differ from the assignment. Their published scores are context, not directly comparable baselines for our new dual-label benchmark.

| Source | What it does | What we can use | What we must not claim as new |
|---|---|---|---|
| Huang et al., 2024 [P1] | Adaptive-threshold CUSUM; fusion of V-I trajectory, power, THD; appliance identification | Event-aligned transient windows, complementary feature groups | Adaptive event thresholds or feature fusion alone |
| Jiang et al., 2024 [P2] | AVME-Hilbert features, excitation attention, semisupervised curriculum labels; noise and low-label tests | A robust feature benchmark and an example of useful ablation studies | Hilbert denoising, attention, or curriculum pseudolabelling alone |
| Le et al., 2021 [P3] | Hilbert amplitude/phase/frequency sequences with Seq2Seq LSTM | An optional Hilbert-feature comparator for transient information | Hilbert features with an LSTM alone |

[P1] reports average appliance-identification F1 of 0.97 and discusses difficulty with unknown loads (Sec. IV; Table V). [P2] reports PLAID accuracy of 96.48%, and 94.17% at 20 dB in its conclusion; its Sec. VI includes denoising, label-ratio, and ablation experiments. [P3] reports PLAID average accuracy 90.04% and F1 94.93% in Sec. IV. These results use their own labels, datasets, and protocols.

**Our defensible difference from these three sources:** a scoped dual-target model coupled with explicit identifiability checks, physics consistency, and an uncertainty-driven acquisition policy tested at equal measurement budgets. This is a proposed combination and evaluation contribution. A limited literature check cannot establish worldwide firstness; active probing, physics-informed learning, and calibration already have broader prior work.

## 4. Data strategy and ground truth

Use controlled simulation as the main source because it provides actual component values and behaviour labels. Python/SciPy is the planned analytical/transient route; ngspice provides an independent circuit check. MATLAB/Simulink is an alternative if already available. No paid tool or cloud GPU is a dependency. Select and pin versions when implementation begins.

**Pilot:** 10 independent parameter configurations per candidate supported pair, including deliberately ambiguous examples. Verify the source, port, controller, settling time, and labels before scaling. **Main target:** about 100-200 base configurations per supported pair, with 3-5 operating-condition episodes each. The final size depends on the audited supported-pair matrix. Count independent configurations separately from windows and noisy copies.

Define a parent physical-configuration ID. All modes, voltage levels, transient crops, and augmented descendants of that physical fixture stay in one split. For parameterised families, prevent effectively identical component settings from appearing across splits; use parameter blocks and a minimum-distance audit, not just different random seeds.

| Field group | Required contents |
|---|---|
| Provenance | configuration ID, episode ID, source/simulator/version, seed, controller family, topology convention |
| Labels | behaviour, composition, validity masks, support-pair ID, reason for ambiguous/unsupported status |
| Physical truth | R/L/C values, parasitics, controller settings, operating region; kept out of model inputs |
| Measurements | timestamps, measured voltage/current, sample rate, source frequency, probe ID, units |
| Conditions | commanded/actual voltage, noise parameters, sensor gain/offset, switching phase, component tolerances |
| Quality | clipping, settling, excitation sufficiency, numerical convergence, missing data |

Freeze **60% training / 15% validation / 10% calibration / 15% test** by physical configuration. Validation selects model/policy hyperparameters. Calibration selects temperatures and reject thresholds. Test is opened once for final evaluation. Noise realisations do not create independent test devices.

PLAID is useful for real-world waveform stress tests, event extraction, or optional pretraining [W2]. It provides appliance measurements, not verified labels for every CC/CP/CZ and RLC combination. Do not relabel refrigerators as RL or laptop supplies as CP without evidence. WHITED is an optional later source after checking its actual metadata and access. Neither is required for the main plan.

## 5. Excitation, parameter ranges, and simulator audit

Start in normalised per-unit simulation, with 50 Hz as the main reference and 60 Hz as a transfer test. The source magnitude is 1.0 pu; physical rescaling is a documented display choice. Initial engineering ranges: R = 5-200 ohm, L = 5-500 mH, C = 5-200 microfarad, sampled approximately log-uniformly. These are candidate ranges; filter by current, voltage, damping, and resonance constraints during the pilot. Use zero for absent ideal components and separately record small numerical/laboratory parasitics.

For each fixture, acquire a nominal waveform, a downward amplitude step, an upward step, and, where needed, a short known multisine or frequency probe. A candidate library uses 0.90/0.95/1.00/1.05/1.10 pu plateaus; the initial deployed schedule uses 1.00, 0.95, and 1.05 pu. Controllers must stay inside their verified regulation limits. Larger variations are stress tests only while labels remain valid.

Record at an initial 10 ksample/s, with an anti-alias bandwidth below Nyquist. Use 20 settled cycles per plateau (0.4 s at 50 Hz), plus measured transition/settling periods. Capture a separate startup/step segment of up to 0.5 s; lengthen it for slow circuits rather than claiming it has settled. The main fixed schedule therefore needs at least 1.2 s plus settling and transitions. Measure acquisition and compute time separately.

Amplitude changes identify behaviour, but do not create independent frequency information for topology. The simulation-only diagnostic probe can add small known 3rd/5th/7th voltage components or use a short frequency sweep. Estimate harmonic impedance only where the voltage excitation is strong enough. When a class remains indistinguishable over the allowed bandwidth, report ambiguity rather than forcing certainty.

Audit every generator using analytical steady-state impedance, energy/passivity checks for passive loads, time-step convergence, and spot checks in ngspice. Verify controller behaviour from terminal measurements, not its mode setting alone. Avoid ideal undamped LC resonance and unstable CP operating points. Use a second controller design or simulator family as a held-out stress test if available, so the model cannot succeed by recognising one generator's artifacts.

## 6. Feature engineering: measure electrical clues correctly

Start with a named feature dictionary of approximately 50-100 scalars. Perform group-wise ablations before adding more. Preserve dimensional and normalised features together; do not normalise each voltage/current trace so aggressively that voltage-response information disappears.

| Feature group | Planned measurements | Why it helps |
|---|---|---|
| Steady state | Vrms, Irms, mean P, S, true PF, fundamental phase and Q1, Z1 | Basic loading, phase, and reactive signature |
| Voltage response | Slopes of log Irms and, where valid, log P against log Vrms; fit residuals; span | Separates CC, CP, CZ and reveals insufficient variation |
| Spectrum | FFT magnitudes/ratios, current/voltage THD, selected complex harmonic impedances | Frequency-dependent and nonlinear signatures |
| Dynamics | Peak current, overshoot, rise/settling time, decay, ringing frequency, transient energy | Distinguishes circuits that look alike in steady state |
| Shape | Crest factor, V-I loop area, signed orientation, correlation | Complementary waveform shape |
| Quality | SNR estimate, clipping/missing masks, excitation span, valid-frequency masks | Prevents unreliable measurements from dominating |
| Optional Hilbert | Narrow-band amplitude/phase/frequency summaries with edge masks | Literature-inspired transient comparator |

Compute P as mean(v(t)i(t)), S as Vrms x Irms, and true PF as P/S. In distorted signals true PF is not generally cos(phi). Define phi = angle(V1) - angle(I1); positive phi and positive Q1 mean lagging/inductive current. Define Q1 = |V1||I1|sin(phi) using RMS fundamental phasors. Report this as fundamental reactive power, not an unspecified universal Q.

Use Z1 = V1/I1 as complex fundamental impedance. Vrms/Irms alone is only a magnitude ratio and does not reveal the full circuit. Avoid division by very small I1 or voltage harmonics; emit validity masks. At zero/near-zero active power, P-based slopes and percentage errors are invalid. Use current/impedance clues instead.

For ideal steady-state behaviour with fixed frequency and approximately fixed PF, nI = d log(Irms)/d log(Vrms) is about 0 for CC, -1 for CP, and +1 for CZ; nP is approximately 1, 0, and 2 respectively. CP's current slope needs the PF assumption; verify active-power constancy separately. Estimate slopes only from settled, distinct voltage points with sufficient span and acceptable fit uncertainty.

For the passive series family, Z(w) = R + j(wL - 1/(wC)), with absent elements omitted. One value of Z cannot uniquely establish all three components. Multiple excited frequencies and transient information are essential. Hilbert instantaneous frequency is not automatically a physical frequency for arbitrary multicomponent signals; filter/select bands, exclude edges, and count any look-ahead delay.

## 7. Baseline model and integrated architecture

The required baseline is a shared MLP: training-fitted feature imputation/scaling -> Dense(96), ReLU, Dropout(0.1) -> Dense(64), ReLU -> shared 64-dimensional embedding -> two separate linear heads of sizes 3 and 7. Train with logits and cross-entropy; apply softmax for display/evaluation. Incomplete labels use per-head masks and normalisation by the number of valid examples for that head.

Initial training settings: AdamW, learning rate 0.001, batch size 128, weight decay 0.0001, up to 100 epochs, early stopping after 12 validation epochs without improvement. These are starting values. Tune a small predeclared grid on validation data; do not repeatedly optimise against the test set. Use five recorded random seeds for final comparisons where feasible.

The deployment graph remains one model with two heads. In a later optional version, a small 1D CNN processes a voltage/current segment; its output joins the scalar branch before a shared 64-dimensional fusion layer. Both heads still read that same fused embedding. Only keep this branch if its ablation justifies the added complexity.

Mandatory comparators are: electrical rules/curve fitting, a random forest on the same features, two separate single-task MLPs, and the integrated baseline. Compare parameter count, training cost, and both-target metrics. Rules are particularly important: clean synthetic voltage slopes may already solve the behaviour task, and the neural model must justify its role on noisy or incomplete observations.

Preprocessing, class order, feature names, training-domain limits, model weights, calibration temperatures, threshold policy, and physical label contract are saved together. The dashboard cannot silently recompute a different scaler or feature definition.

## 8. Enhancement A: electrical consistency and robust fingerprints

**Hypothesis:** adding voltage-response and dynamic features plus valid electrical consistency penalties improves generalisation under sensor noise and unseen parameters compared with the ordinary-feature baseline.

Use a weighted two-head cross-entropy objective. Add a masked behaviour penalty comparing the probability-weighted theoretical current slope with the observed voltage-response slope: expected nI = 0 p(CC) - 1 p(CP) + 1 p(CZ). Use a Huber residual, weighted by excitation quality and slope uncertainty. Disable this term for insufficient voltage span, poor settling, near-zero current, and CP cases where the PF assumption fails. The regression target is measured input-derived evidence, not hidden simulator metadata.

For passive CZ samples only, optionally fit each allowed circuit family to multiple measured complex impedances and obtain a normalised residual dc. Penalise the probability-weighted residual sum over candidate compositions. Fit only using available measurements. Do not apply a passive-series impedance law to an actively controlled terminal. Treat unidentifiable residual ties as ambiguity, not false supervision.

Select physics weights on validation data; start with 0, 0.01, and 0.1, and inspect whether cross-entropy learning degrades. A penalty is useful only if held-out performance improves or invalid confident predictions decrease. It is not a full differential-equation PINN merely because it uses a physics term.

Train with label-preserving domain randomisation: sensor gain/offset, limited phase/timing errors, additive white and coloured noise, quantisation, and missing samples with masks. Recompute features after waveform corruption. Keep true behaviour and topology unchanged. Do not perturb one channel or component in a way that changes the physical target while retaining its old label. Use the same augmentation budget in fairness controls.

Explain results with electrical evidence and grouped permutation importance: voltage-response slope, reactive sign, frequency dependence, and ringing. Group correlated features so that a single arbitrary feature does not receive all credit. These explanations describe evidence and sensitivity, not proof of an appliance's hidden construction.

## 9. Enhancement B: calibrated confidence and adaptive acquisition

**Hypothesis:** uncertainty-guided acquisition improves joint accuracy at a fixed measurement budget, or uses fewer measurements at comparable error, relative to fixed and random acquisition.

Fit one positive temperature per head on the untouched calibration split; temperature scaling is an established method, not our invention [W3]. Report NLL, Brier score, reliability plots, and expected calibration error before/after. Independent head probabilities do not imply independent correctness. Do not multiply them and label the result a calibrated joint probability.

Use a conservative acceptance score such as the minimum of the two calibrated maximum probabilities, plus electrical quality gates. Select the threshold on calibration data for a target accepted-set joint accuracy, initially 95%. Always show coverage and confidence intervals; reject-all is not a success. Calibration under the training distribution does not guarantee reliability on new operating domains.

The policy starts with the nominal observation. If voltage span is insufficient, it must request a voltage step even if the network's softmax is high. After two or more settled voltage points, behaviour may become identifiable. If composition remains ambiguous, request a known harmonic/frequency probe or a better transient. Stop only when the confidence and excitation gates pass, or the fixed maximum budget is exhausted. Otherwise return both tentative labels with an explicit inconclusive status.

Begin with a deterministic policy, using per-head uncertainty and a validation-built table of which probe reduces each ambiguity. No reinforcement learning is necessary. A later expected-information-gain policy is stretch work. During evaluation, select a probe before revealing its stored response: accessing unchosen measurements or true labels would be oracle leakage.

Compare against fixed-order acquisition at every budget, random-order acquisition, and the full-library reference. All policies use the same available probes and trained representation, or explicitly matched retraining. Train with legal missing-probe masks so a policy is not penalised for an input format the model never saw. Plot joint accuracy against number of probes and total acquisition time. An abstaining system also needs risk-versus-coverage plots and complete-system accuracy; conditional accuracy alone is misleading.

All automatic excitation is simulation-only in the required project. A CSV-only real measurement demo may request another file/measurement but cannot invent an unrecorded voltage response. Physical excitation is optional supervised low-voltage lab work, outside the default plan.

## 10. Experimental design and ablations

Freeze an experiment manifest before the final test: data split IDs, feature groups, seeds, architectures, parameter counts, training budgets, hyperparameter candidates, condition grid, and metric definitions. Mark every claimed improvement as supported, unsupported, or inconclusive after measurement.

| ID | Model/variant | Question answered |
|---|---|---|
| R0 | Physics rules / circuit fitting | Is AI needed beyond ideal electrical relationships? |
| C0 | Random forest, same inputs | Is the neural representation competitive? |
| S0 | Separate MLPs | Does sharing help accuracy, size, or compute? |
| B0 | Shared two-head MLP, ordinary features | Required integrated baseline |
| E1 | B0 + voltage/dynamic/frequency fingerprints | Do the new feature groups help? |
| E2 | E1 + realistic noise training | Does robustness training help? |
| E3 | E2 + valid physics penalties | Do electrical constraints add value? |
| E4 | E3 + temperature scaling/rejection | Is confidence more trustworthy? |
| E5 | E4 + adaptive acquisition | Does the measurement policy help? |

Run the core table on identical grouped splits; five seeds for neural variants. Physics/calibration/policy stages may share frozen weights where appropriate. Give B0/E1-E4 the same acquired measurements and training/tuning budget; feature engineering changes the representation. Include a matched-augmentation comparison to isolate the physics term. For E5, report equal-budget and equal-error comparisons, including sensor/acquisition cost. Add the reverse ablation E3 without noise training so one favourable incremental ordering does not hide interactions.

Required test conditions: clean nominal data; additive noise at 40/30/20 dB with 10 dB as a difficult stress case; unseen sensor gain/phase bias; held-out component ranges; source-frequency shift; voltage sag/swell within valid regulation limits; weak excitation; near-resonant or nearly equivalent circuits; and unsupported/mixed ZIP or unseen-controller examples. Report unsupported inputs separately from closed-set 10-class-head scoring.

Measure SNR from clean waveform signal power and injected-noise power; corrupt both channels realistically and document correlations. Some conditions deliberately change the true mode or destabilise a controller. Those examples receive changed/invalid labels or an OOD status rather than silently retaining a wrong target.

If a full-grid run is too costly, keep clean/20 dB/parameter-shift/ambiguity tests plus the full core ablation table. Hardware and heavy paper reproductions are cut first.

## 11. Metrics and the professor's screenshot

The screenshot lists classification and regression statistics together. We will report the useful ones on the quantities they actually measure. Never compute RMSE, MAPE, or R2 on arbitrary numeric encodings such as R = 0 and L = 1.

| Metric | Planned valid use |
|---|---|
| Accuracy, precision, recall, macro-F1 | Each head; per-class support; macro-F1 is primary under imbalance |
| Confusion matrices | Separate 3 x 3 and 7 x 7 matrices; show systematic confusions |
| Joint exact-match accuracy | Both labels correct on samples with both valid labels; also per supported pair |
| ROC-AUC | Macro one-vs-rest using probabilities; define class order and missing-class handling [W4] |
| MAE, MSE, RMSE | Continuous voltage-response slope estimates or a genuine waveform reconstruction experiment |
| R2, RPD | Continuous estimates with enough target variance; RPD = SD(y)/RMSEP, define denominator and units |
| MAPE | Only nonzero continuous targets; undefined near zero, so omit where inappropriate |
| Z-score | Training-fitted feature standardisation, z = (x - mean)/SD; not a classification score |
| Skill score | Brier skill = 1 - BS_model/BS_reference; reference uses training-set class frequencies |
| NLL, Brier, ECE | Probability quality, per head, with documented binning and multiclass convention |
| Coverage/selective risk | Fraction accepted and fraction wrong among accepted; include joint-label risk |
| Latency and resources | Acquisition time; feature/model/UI compute p50/p95; model size, parameters, peak RAM |

The slope estimator is a useful continuous side experiment without adding a third neural output head. Compare noisy two-point and robust multi-point slope estimates against the known ideal exponent only for valid simulated regimes. Use clean signal reconstruction metrics only if an actual denoising stage is implemented. RMSE is a prediction-error measure; neural training convergence is primarily shown by training/validation loss curves.

For Brier score, use the sum of squared probability errors over classes averaged across examples, and use the same convention for the reference. Skill 0 equals the reference, positive is better, and negative is worse. Document undefined metrics instead of replacing them with favourable values. The regression metric RPD is also undefined/uninformative when target variance is zero.

## 12. Statistical comparison and acceptance targets

Report mean and standard deviation over training seeds, plus 95% confidence intervals from resampling independent physical configurations. Bootstrap the entire parent group, including its measurement episodes. Thousands of correlated windows are not thousands of independent test observations. Use paired predictions on the same held-out groups when comparing variants.

Primary evidence is the effect size and group-bootstrap interval for macro-F1/joint-accuracy differences. A paired permutation test at configuration level can support the main comparison. McNemar's test is suitable only with an explicitly defined independent paired correctness unit; correlated windows do not qualify.

To accommodate the professor's requested ANOVA, plan a supplemental repeated-measures comparison across independently generated, balanced device panels. Each panel is a block evaluated by every fixed model; average over seeds within the block. Aim for at least 10 disjoint panels with adequate representation of all evaluated classes, expanding the simulator test set if needed. Check residual assumptions and sphericity; use an appropriate correction and Holm-adjusted post-hoc paired tests. If assumptions fail, use Friedman with a suitable post-hoc procedure and state why. Do not use ordinary independent-groups ANOVA on overlapping cross-validation folds, or count seeds and noise copies as independent devices. This analysis supports conclusions conditional on the tested simulator and trained models.

| Target, not a promised result | Initial acceptance criterion |
|---|---|
| Clean held-out data | Aim for macro-F1 >= 0.95 per head and joint accuracy >= 0.90 |
| Difficult 20 dB condition | Aim for >= 3 percentage-point joint gain over B0, or a clear error reduction with CI |
| Calibration | Lower NLL/Brier; aim ECE <= 0.05 on in-domain test data |
| Accepted predictions | Aim >= 95% joint accuracy at >= 70% coverage; report failures honestly |
| Adaptive acquisition | Aim >= 20% fewer probes/time at comparable error, or better accuracy at equal budget |
| Laptop performance | Aim feature extraction plus model inference p95 < 100 ms per prepared observation; benchmark locally |

If the clean baseline is already near-perfect, prioritise robustness, calibration, measurement cost, and hard ambiguity cases. Do not create an artificially weak baseline to manufacture a gain. Missing a target is a result to analyse, not a reason to hide the experiment.

## 13. Deployment and the professor-facing demonstration

Deploy locally using a saved PyTorch model and a Streamlit dashboard. This is a proposed stack; verify current official installation guidance during implementation. A separate API, paid hosting, and a website framework are unnecessary for the required deliverable. A clean-machine reproduction guide or pinned environment is part of deployment.

Provide two input routes: replay a held-out simulation episode and load a measured CSV with timestamp, voltage, and current columns plus sample rate/units/port metadata. Validate channel alignment, clipping, units, sample rate, required excitation, and available probe masks. Reject unsupported files with a readable explanation. Never treat a single CSV at one constant voltage as enough evidence for CC/CP/CZ.

The dashboard shows: voltage/current waveforms; voltage-response and impedance evidence; head A and B probabilities; tentative/accepted status; a short electrical explanation; the next requested measurement; and baseline-versus-enhanced results. Show the applicable class contract next to composition results. Ground truth can be revealed after prediction in evaluation mode and must not enter inference.

**Four-minute demo script:** (1) a simple CZ-R case; (2) a valid CP controlled fixture whose current rises during a sag; (3) an ambiguous RL/RLC case at one frequency where the system requests more evidence; (4) the same unseen configuration with 20 dB noise, showing the recorded baseline/enhanced comparison; (5) an unsupported example that remains inconclusive. Use only measured benchmark outcomes; do not prefill impressive numbers.

Reproducibility bundle later: model/preprocessor/calibration artifacts, data manifest and simulator specification, experiment configurations, machine-readable metrics, report figures, application instructions, limitations/model card, and a short demo recording. The current folder contains planning documents and source references only.

## 14. Four-week execution schedule and resource plan

Assume one student, an existing laptop, 45-65 focused hours, and four weeks. There is no deadline in the brief, so these are relative weeks, not fabricated submission dates. CPU training is the default for the MLP; a GPU is optional for waveform experiments. Use float32 waveforms, compressed/chunked files, and cache computed features. Estimate disk use from the pilot before expanding the probe library; allow approximately 5-15 GB as an initial budget, subject to actual episode lengths.

| Stage | Time budget | Deliverable and gate |
|---|---:|---|
| Week 1: definitions and pilot | 10-14 h | Agreed taxonomy; supported-pair table; verified simulator and split manifest |
| Week 2: baseline and deployment skeleton | 10-14 h | Rules/RF/MLP results; shared two-head checkpoint; local inference demonstration |
| Week 3: enhancements | 14-20 h | Feature/robustness/physics ablations; calibration; deterministic adaptive policy |
| Week 4: evidence and presentation | 11-17 h | Frozen final evaluation, statistics, dashboard, report, rehearsal and packaging |

Do not postpone baseline deployment until every enhancement is finished. Save a working milestone first. Reserve the final two sessions for reproducibility and presentation; do not introduce a new large architecture at the deadline.

Proposed later repository organisation: simulator and data contracts; feature extraction; shared models and training; evaluation/statistics; dashboard; experiment configurations; data manifests; saved model artifacts; and report outputs. These are future work areas, not folders of implementation code being created now.

**If only one week is available:** retain the definition gate, a verified small simulator, shared MLP, rules/RF comparisons, voltage-response features, 20 dB noise study, basic calibration, and local deployment. Describe adaptive acquisition as implemented only if it actually works; otherwise present it as future work. Prefer three well-tested enhancements to many unfinished additions.

## 15. Marks, risks, and what makes the submission strong

| Mark area | Evidence strategy |
|---|---|
| Development/deployment: 4 | Valid data, shared two-head model, proper held-out evaluation, saved/reloaded live demo |
| Enhancement evidence: suggested 2 | Voltage/dynamic/frequency feature ablations and a strong analytical comparator |
| Enhancement evidence: suggested 2 | Physics penalty with regime masks and an ablation |
| Enhancement evidence: suggested 2 | Noise/operating-condition robustness and unseen-parameter tests |
| Enhancement evidence: suggested 2 | Calibrated confidence, honest inconclusive outputs, risk/coverage plots |
| Enhancement evidence: suggested 2 | Adaptive measurement versus fixed/random policies at matched budgets |
| Enhancement evidence: suggested 1 | Electrical explanations, statistical comparison, reproducible report |

The suggested enhancement rows sum to 11, but the professor decides the actual allocation. Report all implemented contributions together and explain the measured value of each.

| Main risk | Planned response |
|---|---|
| Invalid CC/CP + passive-topology combinations | Agree the fixture/port definition first; audit support; mask missing labels |
| One-frequency ambiguity | Use transient/frequency evidence; include hard examples; allow inconclusive status |
| Simulator shortcuts | Overlap nominal power ranges across classes; randomise nuisance conditions; hold out generator families |
| Inflated test results | Parent-level splitting; train-only scaling; separate calibration; no test tuning |
| Overclaiming topology | Restrict to the declared series family; test unsupported structures separately |
| Physics term hurts controlled loads | Disable unsupported constraints; compare with weight zero |
| Too much scope | Compact MLP first; adaptive rule next; cut heavy denoising/online learning/hardware |
| Deployment hides poor evidence | Show coverage, acquisition delay, failed cases, and real baseline metrics |

The strongest impression will come from a clear problem definition, a working integrated model, a specific tested improvement, and the ability to explain why a difficult example succeeds or fails. A complicated architecture and a single high accuracy number do not establish that.

## 16. Plain-language explanation and viva preparation

Think of the AI as an electrical detective. It has two questions to answer about the same device. First: when the voltage changes, does the device keep current constant, keep power constant, or behave like a fixed impedance? Second: do its measured electrical effects match resistance, inductance, capacitance, or a combination?

Voltage is like pressure and current is like flow, as a rough analogy. A CC device keeps the flow almost steady when pressure changes. A CP device takes more current when voltage drops to preserve its power. A CZ device lets current rise or fall with voltage. Resistance uses energy; an inductor stores it in a magnetic field; a capacitor stores it in an electric field. Inductors and capacitors also change when current peaks relative to voltage and how a circuit responds after a change.

The shared feature extractor is one common set of learned clues. The two output heads are two exam answers written from those same clues. Sharing can reduce model size and can help learning, but we will measure whether it actually helps.

We will first create circuits whose true answers are known. We will observe them, train the ordinary model, then make the observations noisier and more difficult. Our improved system will use electrical relationships, say when it is unsure, and request a useful additional simulated observation. Finally, we will compare the two systems and show the result in a dashboard.

The impressive part is a system that can say: **These first measurements fit more than one circuit. A second measurement separates them, and this experiment shows the improvement.** That is more defensible than claiming it can see hidden components from one waveform.

Prepare short answers to: Why two heads? Why vary voltage? Why is PF not always cos(phi)? Why can RLC look resistive near resonance? Why are CC and CP not passive RLC labels? How was leakage prevented? Which enhancement helped most? What does an inconclusive result mean? Can the system generalise beyond simulation? The final answer to the last question must reflect actual external tests, not an assumption.

## 17. References and evidence boundaries

**[P1]** Yaqian Huang, Yanqing Zhu, Jingyi Pan, Yunpeng Gao, Fenghua Peng, and Yichuang Sun. *Decision Fusion-Based Nonintrusive Load Identification Involving Adaptive Threshold Event Detection*. IEEE Transactions on Instrumentation and Measurement, vol. 73, 2024, article 9004111. DOI: https://doi.org/10.1109/TIM.2024.3413152. Supplied PDF; relevant sections: II, III, IV and Tables V-VI.

**[P2]** Jiahao Jiang, Zhelong Wang, Hongyu Zhao, and Sen Qiu. *Load Identification Based on Attention Semisupervised Curriculum Label Learning With AVME-HT Feature*. IEEE Transactions on Instrumentation and Measurement, vol. 73, 2024, article 6503414. DOI: https://doi.org/10.1109/TIM.2024.3380587. Supplied PDF; relevant sections: I, V-VII, Fig. 12, and ablation discussion.

**[P3]** Thi-Thu-Huong Le, Shinwook Heo, and Howon Kim. *Toward Load Identification Based on the Hilbert Transform and Sequence to Sequence Long Short-Term Memory*. IEEE Transactions on Smart Grid, vol. 12, no. 4, 2021, pp. 3252-3264. DOI: https://doi.org/10.1109/TSG.2021.3066570. Supplied PDF; relevant sections: III-V and Table VI.

**[W1]** EPRI OpenDSS. *Power Conversion Elements*. https://opendss.epri.com/PowerConversionElements.html. Official definitions of constant-P/Q, constant-Z, constant-I, and ZIP models. Checked 20 October 2026.

**[W2]** Roberto Medico et al. *A voltage and current measurement dataset for plug load appliance identification in households*, 2020. Author-hosted paper: https://ugentai4e.github.io/papers/2020/medico2020scidata.pdf. Supports PLAID's waveform/metadata role; does not establish our dual-label ground truth. Checked 20 October 2026.

**[W3]** Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. *On Calibration of Modern Neural Networks*. ICML, PMLR 70, 2017, pp. 1321-1330. https://proceedings.mlr.press/v70/guo17a.html. Supports temperature scaling as an established calibration method. Checked 20 October 2026.

**[W4]** scikit-learn. *Metrics and scoring: quantifying the quality of predictions*. https://scikit-learn.org/stable/modules/model_evaluation.html. Official metric definitions and multiclass ROC-AUC guidance. Checked 20 October 2026.

**[S1]** Supplied professor-message screenshot, copied as `References/Professor_metric_notes.png`. Used as contextual evidence for the metric discussion, without treating it as an instruction to misuse regression statistics.

Everything described as an architecture, policy, parameter range, target score, time budget, or enhancement is a proposal for later implementation. No experimental result, globally novel discovery, deployment, professor approval, or grade is claimed by this plan.
