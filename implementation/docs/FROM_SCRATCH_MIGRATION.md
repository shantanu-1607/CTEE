# From-scratch migration: checkpoint 1

**Requirement:** the model and its learning algorithms must be implemented explicitly. NumPy is allowed for numerical calculations. Git commits and pushes are performed by the project owner, without assistant co-author attribution.

## What is implemented in this checkpoint

`ctee/scratch/` contains a standalone shared neural network with a three-class behaviour head and a seven-class composition head. Both tasks update the same hidden layers. This package imports only NumPy and standard Python; it does not use PyTorch, TensorFlow, scikit-learn, SciPy, an automatic-differentiation engine, or a prebuilt optimizer.

Implemented explicitly:

- Dense-layer forward calculations and shared-encoder backpropagation.
- ReLU, leaky ReLU, and tanh activations and derivatives.
- He/Xavier initialisation and inverted dropout, disabled during inference.
- Stable softmax and weighted cross-entropy with label smoothing and per-head label masks.
- An optional Huber slope-consistency loss with its softmax derivative.
- Additive measured-evidence priors as optional input logits; priors must come from available measurements, not hidden labels.
- Adam moment updates, bias correction, epsilon, decoupled weight decay, and global gradient clipping.
- SGD, momentum, and optional Nesterov updates.
- Training-only mean/scale fitting, mini-batches, constant/cosine learning rates, validation-based early stopping, and best-weight restoration.
- Safe numerical-array checkpoints using NPZ and JSON metadata with `allow_pickle=False`.

This is an inference checkpoint format. Resuming the exact optimizer/RNG state is not implemented in this batch.

## Parameters we can now experiment with

| Area | Exposed controls |
|---|---|
| Encoder | Number and width of hidden layers, activation, leaky slope, initialisation, dropout, numeric precision |
| Inputs | Scaling floor and normalised-input clipping |
| Objective | Behaviour/composition weights, class weights, label smoothing, L2 coefficient, slope-consistency weight and Huber delta |
| Evidence | Additive prior strength; caller supplies only measured evidence |
| Adam | Learning rate, beta1, beta2, epsilon, decoupled weight decay, clipping norm |
| SGD | Learning rate, momentum, Nesterov, decoupled weight decay, clipping norm |
| Training | Batch size, epochs, patience, minimum improvement, random seed, learning-rate schedule |

L2 changes the objective gradient. Decoupled weight decay changes the optimizer update. They are separate configurable controls; both default model-objective L2 and evidence/physics terms are zero in this standalone core. Decay applies to weight matrices, not biases.

## Verification

Run from `implementation/`:

```sh
.venv/bin/python -m pytest tests/test_scratch.py -q
```

The tests check every trainable parameter's derivative using centred finite differences, including weighted/smoothed masked losses and physics consistency. They also verify dropout gradients with a fixed random mask, that the two heads' encoder gradients add, hand-calculated optimizer steps, gradient clipping, checkpoint equivalence, early-stop restoration, and learning both tasks on a small numerical sanity dataset. That sanity dataset is not an electrical-load benchmark and its score must not be reported as project accuracy.

## What is still library-based

The old `ctee/model.py`, `ctee/train.py`, `ctee/metrics.py`, `ctee/evaluate.py`, and dashboard still run the earlier PyTorch/scikit-learn pipeline. The existing 100% clean and 96.3% noisy benchmark scores belong to that older implementation. They are **not** scores for this new from-scratch network.

The signal-generation and feature modules still use SciPy routines for the startup solver, filtering, and constrained fitting, and NumPy high-level numerical routines for signal fitting. These need a separate decision/migration if the professor's restriction also covers signal-processing solvers. Plotting/UI/file tools do not implement the classifier, but their permitted use should be documented for the final submission. The current project as a whole is therefore **not yet a fully migrated from-scratch submission**.

## Next small batch, after this checkpoint is committed

1. Connect this core to the group-separated electrical dataset with explicit evidence masks.
2. Provide editable experiment configurations and a manual classification-metric implementation.
3. Train new baseline/enhanced models and record their own validation results in a separate location.
4. Migrate calibration, inference, and the dashboard after verifying parity and correctness.
5. Re-run the frozen comparisons on genuinely held-out data. Do not tune against previously published test results and call them a new blind test.

Each batch stops with a concise change/test summary and commit/push commands. Model quality and novelty must be demonstrated experimentally; neither is guaranteed by implementing the arithmetic manually.
