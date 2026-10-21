# CTEE project: start here

**Status: planning only. Prepared 20 October 2026. No simulator, dataset, model, training, or application has been implemented.**

## The project in one sentence

Build one AI system that looks at voltage and current and answers two questions: **how does the load behave when voltage changes, and what kind of R/L/C circuit signature does it have?**

## What the two answers mean

| Question | Possible answers | Everyday meaning |
|---|---|---|
| How does it respond to voltage? | CC, CP, CZ | Does it try to keep current, power, or impedance constant? |
| What circuit does it resemble? | R, L, C, RL, RC, LC, RLC | Does it mainly dissipate energy, store it magnetically/electrically, or combine these effects? |

R is resistance, L is inductance, and C is capacitance. A real appliance can have electronic controls, so its operating behaviour and its passive components are different descriptions. The detailed plan makes that distinction explicit.

## A simple example

Imagine a load using 100 W at its usual voltage, with power factor approximately one. Reduce the voltage by 10%:

| Behaviour | What happens? | Approximate power |
|---|---|---:|
| CC | Current stays the same | 90 W |
| CP | Current increases to preserve power | 100 W |
| CZ | Current falls along with voltage | 81 W |

These are idealised examples over a valid operating range. They explain why we need measurements at more than one voltage. At one voltage, the three loads could look identical.

## What we will actually do later

1. Agree precise class definitions and simulate circuits with known answers.
2. Record their voltage/current waveforms under several conditions.
3. Extract useful clues: RMS, phase, power, harmonics, impedance, voltage-response slopes, and transient behaviour.
4. Train a simple shared neural network with two output heads as the baseline.
5. Add our proposed improvement: electrical consistency checks, robustness to measurement noise, calibrated confidence, and an extra simulated measurement when evidence is insufficient.
6. Compare the improved model against the baseline on unseen circuits and difficult conditions.
7. Deploy the trained model in a local dashboard with waveform plots, both predictions, an explanation, and baseline comparisons.

## What makes the project interesting

An ordinary classifier guesses from the measurements it receives. Our proposed system also asks: **Do these measurements contain enough information? Do the answers agree with electrical laws? Which extra measurement would help most?**

The proposed contribution is *uncertainty-guided electrical fingerprinting*. It is a project-level research hypothesis, not a claim that nobody in the world has used these ideas before. The experiments must show whether it improves reliability or reduces measurement time.

## Read these files in this order

- `01_IMPLEMENTATION_PLAN.pdf`: the main readable plan, including architecture, data, novelty, experiments, timeline, and an easy explanation.
- `01_IMPLEMENTATION_PLAN.md`: the editable source of the main plan.
- `02_EXECUTION_CHECKLIST.md`: ordered tasks and completion gates for the later implementation.
- `03_PROFESSOR_DISCUSSION.md`: a short proposal and the important scope questions to take to your professor.
- `References/`: copies of your three papers and the supplied screenshot. The bibliography and literature comparison are in the main plan.

## Working assumptions

One student, an existing laptop, approximately four weeks and 45-65 focused hours, and simulation as the required demonstration. Physical hardware is optional. Change the schedule when the actual deadline and lab access are known.

The first decision is the meaning of circuit composition for an electronically controlled CC/CP load. The recommended interpretation is a documented passive input network plus a controller; this needs professor agreement before data generation. We will not manufacture 21 arbitrary class combinations or pretend that all are identifiable.

No grade can be guaranteed. This plan is designed to produce clear evidence for both the 4-mark implementation requirement and the 11-mark enhancement requirement.
