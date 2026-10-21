"""A shared MLP, its two output heads, and their analytical derivatives.

No neural-network framework, automatic differentiation, or library estimator.
Weights use the convention (input units, output units).
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class NetworkConfig:
    input_dim: int
    hidden_widths: tuple = (96, 64)
    activation: str = "relu"
    negative_slope: float = 0.01
    dropout: float = 0.1
    initialization: str = "he"
    input_clip: float = 15.0
    scale_floor: float = 0.01
    prior_strength: float = 0.0
    learned_physics_gate: bool = False
    gate_feature_indices: tuple = ()
    dtype: str = "float64"
    seed: int = 11

    def __post_init__(self):
        if not isinstance(self.input_dim, int) or self.input_dim < 1:
            raise ValueError("input_dim must be a positive integer")
        if not self.hidden_widths or any(not isinstance(w, int) or w < 1 for w in self.hidden_widths):
            raise ValueError("Choose at least one positive integer hidden width")
        if self.activation not in {"relu", "leaky_relu", "tanh"}:
            raise ValueError("activation must be relu, leaky_relu, or tanh")
        if self.initialization not in {"he", "xavier"}:
            raise ValueError("initialization must be he or xavier")
        values = [self.dropout, self.negative_slope, self.input_clip, self.scale_floor, self.prior_strength]
        if not np.isfinite(values).all():
            raise ValueError("Network settings must be finite")
        if not 0 <= self.dropout < 1 or not 0 <= self.negative_slope < 1:
            raise ValueError("dropout and negative_slope must be in [0, 1)")
        if self.input_clip <= 0 or self.scale_floor <= 0 or self.prior_strength < 0:
            raise ValueError("Invalid clipping, scaling, or prior strength")
        if not isinstance(self.learned_physics_gate, bool):
            raise ValueError("learned_physics_gate must be boolean")
        if self.learned_physics_gate and self.prior_strength <= 0:
            raise ValueError("A learned physics gate requires a positive maximum prior strength")
        if len(set(self.gate_feature_indices)) != len(self.gate_feature_indices) or any(
                not isinstance(i, int) or not 0 <= i < self.input_dim for i in self.gate_feature_indices):
            raise ValueError("Gate feature indices must be unique input indices")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")


def _finite_matrix(x):
    x = np.asarray(x)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Expected a nonempty finite matrix")
    return x


def softmax(logits):
    z = _finite_matrix(logits)
    shifted = z - z.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _mask(mask, n):
    if mask is None:
        return np.ones(n, dtype=bool)
    result = np.asarray(mask)
    if result.shape != (n,) or result.dtype != np.bool_:
        raise ValueError("Masks must be boolean vectors matching the batch")
    return result


def cross_entropy(logits, labels, mask=None, smoothing=0.0, class_weights=None):
    """Return mean observed-label CE and d(loss)/d(logits).

    Masked labels can be -1. Weights use the true class, including when label
    smoothing is enabled; normalisation is by the observed sample-weight sum.
    """
    z = _finite_matrix(logits)
    n, k = z.shape
    y = np.asarray(labels)
    if y.shape != (n,) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError("Labels must be an integer vector matching the batch")
    observed = _mask(mask, n)
    if not 0 <= smoothing < 1:
        raise ValueError("smoothing must be in [0, 1)")
    if ((y[observed] < 0) | (y[observed] >= k)).any():
        raise ValueError("Observed label outside the head's class range")
    weights = np.ones(k) if class_weights is None else np.asarray(class_weights, dtype=float)
    if weights.shape != (k,) or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("class_weights must contain a positive finite weight for each class")
    grad = np.zeros_like(z)
    ids = np.flatnonzero(observed)
    if not len(ids):
        return 0.0, grad
    shifted = z[ids] - z[ids].max(axis=1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    target = np.full((len(ids), k), smoothing / k)
    target[np.arange(len(ids)), y[ids]] += 1 - smoothing
    sample_weights = weights[y[ids]]
    normalizer = sample_weights.sum()
    loss = -np.sum(sample_weights[:, None] * target * logp) / normalizer
    grad[ids] = (np.exp(logp) - target) * sample_weights[:, None] / normalizer
    return float(loss), grad


def slope_consistency(logits, slopes, mask=None, delta=1.0):
    """Huber penalty for E[nI] with the ideal CC/CP/CZ slopes [0, -1, 1].

    The caller must mask regimes with insufficient excitation or invalid CP
    power-factor assumptions. This function does not decide physical validity.
    """
    z = _finite_matrix(logits)
    if z.shape[1] != 3 or not np.isfinite(delta) or delta <= 0:
        raise ValueError("Slope consistency needs three classes and positive Huber delta")
    target = np.asarray(slopes)
    if target.shape != (len(z),):
        raise ValueError("Slope targets must match the batch")
    observed = _mask(mask, len(z))
    if not np.isfinite(target[observed]).all():
        raise ValueError("Observed slope targets must be finite")
    ids = np.flatnonzero(observed)
    grad = np.zeros_like(z)
    if not len(ids):
        return 0.0, grad
    p = softmax(z[ids])
    theory = np.array([0.0, -1.0, 1.0])
    expected = p @ theory
    error = expected - target[ids]
    absolute = np.abs(error)
    losses = np.where(absolute <= delta, 0.5 * error**2, delta * (absolute - 0.5 * delta))
    d_expected = np.clip(error, -delta, delta)
    grad[ids] = d_expected[:, None] * p * (theory[None, :] - expected[:, None]) / len(ids)
    return float(losses.mean()), grad


class TwoHeadMLP:
    """An explicit shared encoder and heads with exactly 3 and 7 classes."""

    def __init__(self, config):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.params = {}
        dimensions = (config.input_dim, *config.hidden_widths)
        for i, (fan_in, fan_out) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            factor = np.sqrt(2.0 / fan_in) if config.initialization == "he" else np.sqrt(2.0 / (fan_in + fan_out))
            self.params[f"W{i}"] = (self.rng.standard_normal((fan_in, fan_out)) * factor).astype(config.dtype)
            self.params[f"b{i}"] = np.zeros(fan_out, dtype=config.dtype)
        for name, classes in [("behaviour", 3), ("composition", 7)]:
            self.params[f"W_{name}"] = (self.rng.standard_normal((dimensions[-1], classes)) / np.sqrt(dimensions[-1])).astype(config.dtype)
            self.params[f"b_{name}"] = np.zeros(classes, dtype=config.dtype)
        if config.learned_physics_gate:
            gate_width = len(config.gate_feature_indices) or config.input_dim
            for head in ("behaviour", "composition"):
                self.params[f"W_gate_{head}"] = np.zeros((gate_width, 1), dtype=config.dtype)
                self.params[f"b_gate_{head}"] = np.zeros(1, dtype=config.dtype)
        self.mean = np.zeros(config.input_dim, dtype=config.dtype)
        self.scale = np.ones(config.input_dim, dtype=config.dtype)

    def fit_scaler(self, training_inputs):
        x = self._inputs(training_inputs)
        self.mean = x.mean(axis=0)
        # Population standard deviation, fitted only on caller-supplied training inputs.
        self.scale = np.maximum(np.sqrt(np.mean((x - self.mean)**2, axis=0)), self.config.scale_floor)

    def _inputs(self, x):
        x = _finite_matrix(np.asarray(x, dtype=self.config.dtype))
        if x.shape[1] != self.config.input_dim:
            raise ValueError("Input feature count differs from the model configuration")
        return x

    def _activation(self, z):
        if self.config.activation == "tanh":
            return np.tanh(z)
        slope = self.config.negative_slope if self.config.activation == "leaky_relu" else 0.0
        return np.where(z > 0, z, slope * z)

    def _activation_derivative(self, z):
        if self.config.activation == "tanh":
            return 1 - np.tanh(z)**2
        slope = self.config.negative_slope if self.config.activation == "leaky_relu" else 0.0
        return np.where(z > 0, 1.0, slope)

    def _forward(self, x, training, priors):
        a = np.clip((self._inputs(x) - self.mean) / self.scale, -self.config.input_clip, self.config.input_clip)
        gate_input = a[:, self.config.gate_feature_indices] if self.config.gate_feature_indices else a
        gates = {}
        layers = []
        for i in range(len(self.config.hidden_widths)):
            z = a @ self.params[f"W{i}"] + self.params[f"b{i}"]
            drop = np.ones_like(z)
            if training and self.config.dropout:
                keep = 1 - self.config.dropout
                drop = (self.rng.random(z.shape) < keep).astype(z.dtype) / keep
            layers.append((a, z, drop))
            a = self._activation(z) * drop
        logits = {head: a @ self.params[f"W_{head}"] + self.params[f"b_{head}"] for head in ("behaviour", "composition")}
        if self.config.learned_physics_gate and priors is None:
            raise ValueError("Learned physics gates require measured prior logits")
        if priors is not None:
            if set(priors) != set(logits):
                raise ValueError("Provide behaviour and composition prior logits together")
            for head in logits:
                p = np.asarray(priors[head])
                if p.shape != logits[head].shape or not np.isfinite(p).all():
                    raise ValueError("Prior logits must match each head and be finite")
                gate = np.ones((len(a), 1))
                if self.config.learned_physics_gate:
                    q = gate_input @ self.params[f"W_gate_{head}"] + self.params[f"b_gate_{head}"]
                    # Stable sigmoid, with no clipped derivative mismatch.
                    gate = np.exp(-np.logaddexp(0., -q))
                gates[head] = gate
                logits[head] = logits[head] + self.config.prior_strength * gate * p
        return logits, (layers, a, gate_input, gates)

    def forward(self, x, priors=None):
        """Deterministic inference: dropout is disabled and no state is updated."""
        return self._forward(x, False, priors)[0]

    def physics_gate_values(self, x):
        """Learned evidence trust in [0,1], independent of class labels and priors."""
        a = np.clip((self._inputs(x) - self.mean) / self.scale, -self.config.input_clip, self.config.input_clip)
        a = a[:, self.config.gate_feature_indices] if self.config.gate_feature_indices else a
        return {head: (np.exp(-np.logaddexp(0., -(a @ self.params[f"W_gate_{head}"] + self.params[f"b_gate_{head}"]))).ravel()
                       if self.config.learned_physics_gate else np.ones(len(a)))
                for head in ("behaviour", "composition")}

    def predict_proba(self, x, priors=None):
        return {name: softmax(z) for name, z in self.forward(x, priors).items()}

    def loss_and_gradients(self, x, yb, yc, *, behaviour_mask=None, composition_mask=None,
                           behaviour_weight=1.0, composition_weight=1.0, smoothing=0.0,
                           behaviour_class_weights=None, composition_class_weights=None,
                           l2=0.0, physics_weight=0.0, slopes=None, slope_mask=None,
                           huber_delta=1.0, training=True, priors=None):
        settings = [behaviour_weight, composition_weight, l2, physics_weight]
        if not np.isfinite(settings).all() or min(settings) < 0 or behaviour_weight + composition_weight <= 0:
            raise ValueError("Loss weights must be finite and nonnegative, with a supervised task enabled")
        logits, (layers, hidden, gate_input, gates) = self._forward(x, training, priors)
        lb, db = cross_entropy(logits["behaviour"], yb, behaviour_mask, smoothing, behaviour_class_weights)
        lc, dc = cross_entropy(logits["composition"], yc, composition_mask, smoothing, composition_class_weights)
        db *= behaviour_weight
        dc *= composition_weight
        loss = behaviour_weight * lb + composition_weight * lc
        physics_loss = 0.0
        if physics_weight:
            if slopes is None:
                raise ValueError("Provide measured slopes when enabling physics consistency")
            physics_loss, dp = slope_consistency(logits["behaviour"], slopes, slope_mask, huber_delta)
            db += physics_weight * dp
            loss += physics_weight * physics_loss
        grads = {}
        d_hidden = np.zeros_like(hidden)
        for head, dlogits in [("behaviour", db), ("composition", dc)]:
            grads[f"W_{head}"] = hidden.T @ dlogits
            grads[f"b_{head}"] = dlogits.sum(axis=0)
            # Both tasks contribute to the same encoder gradient.
            d_hidden += dlogits @ self.params[f"W_{head}"].T
            if self.config.learned_physics_gate:
                dgate = self.config.prior_strength * np.sum(dlogits * priors[head], axis=1, keepdims=True)
                dq = dgate * gates[head] * (1 - gates[head])
                grads[f"W_gate_{head}"] = gate_input.T @ dq
                grads[f"b_gate_{head}"] = dq.sum(axis=0)
        for i in reversed(range(len(layers))):
            previous, z, drop = layers[i]
            dz = d_hidden * drop * self._activation_derivative(z)
            grads[f"W{i}"] = previous.T @ dz
            grads[f"b{i}"] = dz.sum(axis=0)
            d_hidden = dz @ self.params[f"W{i}"].T
        for name, value in self.params.items():
            if name.startswith("W"):
                loss += 0.5 * l2 * float(np.sum(value**2))
                grads[name] += l2 * value
        details = {"loss": float(loss), "behaviour_ce": lb, "composition_ce": lc, "physics_huber": physics_loss}
        return details, grads

    def save(self, path, training_metadata=None):
        """Array checkpoint with JSON metadata; no pickle or framework format."""
        path = Path(path)
        if path.suffix != ".npz":
            raise ValueError("Scratch checkpoints use the .npz extension")
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = json.dumps({"format": "ctee_numpy_v1", "config": asdict(self.config), "training": training_metadata or {}})
        np.savez_compressed(path, **self.params, mean=self.mean, scale=self.scale, metadata=np.array(meta))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(saved["metadata"].item())
            if metadata.get("format") != "ctee_numpy_v1":
                raise ValueError("Unrecognised scratch checkpoint format")
            config = metadata["config"]
            config["hidden_widths"] = tuple(config["hidden_widths"])
            config["gate_feature_indices"] = tuple(config.get("gate_feature_indices", ()))
            model = cls(NetworkConfig(**config))
            if set(saved.files) != set(model.params) | {"mean", "scale", "metadata"}:
                raise ValueError("Checkpoint fields do not match this architecture")
            for name, expected in model.params.items():
                value = saved[name]
                if value.shape != expected.shape or not np.isfinite(value).all():
                    raise ValueError("Invalid checkpoint parameter")
                model.params[name] = value.astype(model.config.dtype, copy=True)
            for name in ("mean", "scale"):
                value = saved[name]
                if value.shape != (model.config.input_dim,) or not np.isfinite(value).all():
                    raise ValueError("Invalid checkpoint scaler")
                setattr(model, name, value.astype(model.config.dtype, copy=True))
            if (model.scale <= 0).any():
                raise ValueError("Checkpoint scales must be positive")
        return model
