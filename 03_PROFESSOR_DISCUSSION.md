# Proposal to discuss with the professor

**Working title:** Uncertainty-Guided, Physics-Consistent Multi-Task Electrical Load Classification

I propose a shared neural feature extractor with two classification heads: CC/CP/CZ and R/L/C/RL/RC/LC/RLC. The model will use measured voltage/current features, voltage-response slopes, harmonic impedance, and transient behaviour. It will be deployed as a local dashboard.

The proposed enhancement is a system that checks whether the available electrical measurements are informative and consistent, estimates calibrated confidence, and requests an additional simulated voltage/frequency measurement when necessary. I will validate this using baseline and ablation comparisons, noise and parameter-shift tests, and accuracy-versus-measurement-cost curves.

The project will explicitly distinguish the voltage-response behaviour of a complete controlled load from its passive input network. Fixed passive RLC circuits are normally CZ at a fixed frequency; arbitrary combinations of CC/CP and pure passive circuits would be misleading. My recommended main benchmark uses a declared series input-network convention, known controllers where regulation is required, and only physically verified supported pairs.

## Questions to resolve before implementation

1. Does circuit composition mean the complete passive load, a known passive input network before a controller, or an equivalent measured RLC signature? May I use the input-network interpretation for regulated CC/CP fixtures?
2. Is a restricted matrix of physically supported behaviour/composition pairs acceptable, instead of assuming all 21 combinations? How should genuinely unidentifiable/unsupported examples be reported?
3. Is controlled simulation with a deployed local inference dashboard sufficient, or is a physical hardware experiment compulsory?
4. What is the submission deadline, report/demo format, team rule, and expected use of the listed regression metrics and ANOVA?

## Proposed evaluation

Primary: macro-F1 and confusion matrices per head, joint exact-match accuracy, robustness under 20 dB noise, calibration, selective risk/coverage, and acquisition/inference time. Continuous metrics will be used for continuous slope estimation or genuine reconstruction tasks. ANOVA will use a suitable repeated-measures design with independent device panels and assumption checks, with a justified alternative if needed.

## Proposed contribution statement

The contribution is an experimentally validated combination of physical fingerprints, regime-aware consistency, and uncertainty-guided measurement for this scoped dual-target task. I will credit the supplied Hilbert/AVME/decision-fusion work and avoid claiming that attention, Hilbert transforms, calibration, or active probing are individually new inventions.

**Requested outcome from the discussion:** agree the label contract and deployment scope, then proceed with the pilot and baseline before implementing enhancements.
