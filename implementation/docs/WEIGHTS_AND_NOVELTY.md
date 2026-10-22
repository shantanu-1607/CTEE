# ML weights, learning, and the proposed contribution

## Yes — we train the actual ML parameters

The original NumPy implementation already learned the encoder matrices, encoder biases, and both heads' matrices/biases. They are arrays in `TwoHeadMLP.params`, not preset circuit answers. We manually compute each derivative and update each array through Adam/SGD. The learning rate, layer widths and optimizer settings are **hyperparameters**; `W0`, `b0`, `W1`, `b1`, `W_behaviour`, `b_behaviour`, `W_composition`, `b_composition` are **learned parameters**.

For a layer, `H = activation(XW + b)`. During backpropagation, `dW = Xᵀ dH`, `db = sum(dH)`, and both heads send gradients into the shared encoder. Adam stores first/second moments, corrects their initialization bias, then changes the weights. Finite differences independently verify the gradients of every parameter, including the physics loss and new gates.

The **Weights and learning** dashboard tab shows the actual arrays, the initial-to-learned differences, training gradient/update norms and a CSV export of every value. An isolated single-coordinate edit shows how predictions change. That preview does not overwrite the deployment checkpoint or claim the old benchmark remains valid after editing.

## Proposed extension: quality-conditioned physics trust

A fixed prior always contributes the same amount of electrical evidence. That can be unhelpful when noisy or misaligned sensors distort the inferred impedance. We add one trainable sigmoid gate per head:

```
q = standardized measured quality indicators
alpha_head = sigmoid(q @ W_gate_head + b_gate_head)
final_logits_head = neural_logits_head + max_strength * alpha_head * physics_logits_head
```

The quality vector contains probe-availability flags, voltage span, slope/spectral validity, circuit-fit residual, FFT high-frequency ratio/entropy and startup availability. It contains no hidden R/L/C values or true class labels. Gates learn only from training examples. They are soft evidence weights, not calibrated probabilities that physics is correct.

The new gate parameters receive exact analytical gradients:

```
d_alpha = max_strength * sum(d_logits * physics_logits)
d_gate_logit = d_alpha * alpha * (1-alpha)
d_W_gate = q.T @ d_gate_logit
d_b_gate = sum(d_gate_logit)
```

The tests compare these gradients with numerical finite differences, verify that deliberately wrong physics can be downweighted through learning, and check that undeclared input features cannot affect a gate.

## Weight averaging

Optional EMA tracks the actual trainable arrays after every optimizer step:

```
ema_weight = decay * ema_weight + (1-decay) * current_weight
```

Both raw and averaged checkpoints are evaluated on validation data. The better validation-loss checkpoint is retained; final family/candidate selection also uses validation data only. Deployment remains one shared encoder and two heads, with no ensemble inference cost. The chosen weight source and update evidence are recorded. EMA does not guarantee improvement and is not a new invention.

## Experiment discipline

The first validation-only study is archived under `runs/numpy_v2/`. A matched follow-up with EMA is under `runs/numpy_v3/`, on the same fresh physical split, before those test circuits have been evaluated. Each family receives three candidate settings, the same waveform randomization, the same stopping rules and three training seeds after candidate selection.

Families are ordinary-feature baseline, features-only, fixed physics prior/consistency, and learned physics gates. All use manual learning. A frozen-weight counterfactual replaces trained gate logits with zero (gate=0.5) without changing neural weights, testing sensitivity to the learned gates. A separate set of 700 new simulator configurations is reserved for final confirmation. If later tuning uses test outcomes, it needs a new final test pool.

Deployment uses the strongest validation checkpoint, even if the experimental gate does not win. Report failed or inconclusive comparisons. A good scientific experiment does not require every new module to beat every baseline.

## What is and is not the novelty claim?

Our proposed project contribution is a **measurement-quality-conditioned electrical-prior mechanism for a shared CC/CP/CZ and RLC classifier, coupled to causal adaptive measurement**, implemented and differentiated manually and evaluated with matched ablations. The parameter inspector provides a reproducible demonstration of how it learns.

Gating, physics-guided learning, exponential weight averaging, calibration and active probing already exist. For example:

- [Shazeer et al., 2017, sparsely gated mixture of experts](https://arxiv.org/abs/1701.06538): learned gating is established; our small dense physics gates are not that architecture.
- [Non-Intrusive Load Monitoring Based on the Combination of Gate-Transformer and CNN, 2023](https://www.mdpi.com/2079-9292/12/13/2824): gating has already appeared in load-monitoring research. A gating layer alone cannot support a first-ever NILM claim.
- [Morales-Brotons, Vogels and Hendrikx, EMA of weights, 2024](https://arxiv.org/abs/2411.18704): EMA is an established method. Our implementation averages parameters explicitly and tests it on this task; it does not reproduce every experiment in that paper.
- [Izmailov et al., weight averaging, 2018](https://proceedings.mlr.press/r16/izmailov18a.html): broader weight-averaging context. Our EMA recurrence is not the same as their SWA schedule.

The supplied Hilbert/AVME/decision-fusion papers remain credited in the original plan. Their appliance labels differ from this declared dual-target series-branch benchmark. The present literature search is limited and does not establish worldwide uniqueness, patent novelty, a globally best classifier, or guaranteed marks.
