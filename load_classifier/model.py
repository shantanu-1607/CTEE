"""A shared ReLU MLP with behaviour and circuit classification heads."""
import json
from pathlib import Path

import numpy as np

from .data import BEHAVIOURS, CIRCUITS


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def cross_entropy(logits, targets):
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    loss = -log_probabilities[np.arange(len(targets)), targets].mean()
    gradient = np.exp(log_probabilities)
    gradient[np.arange(len(targets)), targets] -= 1
    return float(loss), gradient / len(targets)


class TwoHeadMLP:
    def __init__(self, features, hidden_sizes=(64, 32), seed=42):
        self.features = tuple(features)
        self.hidden_sizes = tuple(hidden_sizes)
        if not self.features or not self.hidden_sizes or any(size < 1 for size in self.hidden_sizes):
            raise ValueError("The network needs inputs and positive hidden layer sizes")
        rng = np.random.default_rng(seed)
        dimensions = (len(self.features), *self.hidden_sizes)
        self.params = {}
        for i, (width, next_width) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            self.params[f"W{i}"] = rng.normal(0, np.sqrt(2 / width), (width, next_width))
            self.params[f"b{i}"] = np.zeros(next_width)
        for head, classes in (("behaviour", 3), ("circuit", 7)):
            self.params[f"W_{head}"] = rng.normal(0, 1 / np.sqrt(dimensions[-1]), (dimensions[-1], classes))
            self.params[f"b_{head}"] = np.zeros(classes)
        self.mean = np.zeros(len(self.features))
        self.scale = np.ones(len(self.features))

    def _check_inputs(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or not len(x) or x.shape[1] != len(self.features) or not np.isfinite(x).all():
            raise ValueError("Expected finite measurement rows matching the model's feature order")
        return x

    def fit_scaler(self, training_inputs):
        x = self._check_inputs(training_inputs)
        self.mean = x.mean(axis=0)
        scale = x.std(axis=0)
        self.scale = np.where(scale > 1e-12, scale, 1.)

    def _forward(self, x):
        hidden = (self._check_inputs(x) - self.mean) / self.scale
        layers = []
        for i in range(len(self.hidden_sizes)):
            linear = hidden @ self.params[f"W{i}"] + self.params[f"b{i}"]
            layers.append((hidden, linear))
            hidden = np.maximum(linear, 0)
        logits = {head: hidden @ self.params[f"W_{head}"] + self.params[f"b_{head}"]
                  for head in ("behaviour", "circuit")}
        return logits, hidden, layers

    def predict_proba(self, x):
        return {head: softmax(logits) for head, logits in self._forward(x)[0].items()}

    def loss(self, x, behaviour, circuit):
        logits = self._forward(x)[0]
        return sum(cross_entropy(logits[head], target)[0]
                   for head, target in (("behaviour", behaviour), ("circuit", circuit)))

    def loss_and_gradients(self, x, behaviour, circuit, l2=0.):
        logits, hidden, layers = self._forward(x)
        gradients = {}
        hidden_gradient = np.zeros_like(hidden)
        loss = 0.
        for head, target in (("behaviour", behaviour), ("circuit", circuit)):
            head_loss, derivative = cross_entropy(logits[head], target)
            loss += head_loss
            gradients[f"W_{head}"] = hidden.T @ derivative
            gradients[f"b_{head}"] = derivative.sum(axis=0)
            hidden_gradient += derivative @ self.params[f"W_{head}"].T
        for i in reversed(range(len(layers))):
            previous, linear = layers[i]
            derivative = hidden_gradient * (linear > 0)
            gradients[f"W{i}"] = previous.T @ derivative
            gradients[f"b{i}"] = derivative.sum(axis=0)
            hidden_gradient = derivative @ self.params[f"W{i}"].T
        for name, parameter in self.params.items():
            if name.startswith("W"):
                loss += .5 * l2 * np.sum(parameter ** 2)
                gradients[name] += l2 * parameter
        return float(loss), gradients

    def save(self, path, metadata):
        path = Path(path)
        info = {"format": "two_head_mlp_v1", "features": self.features,
                "hidden_sizes": self.hidden_sizes, "behaviours": BEHAVIOURS,
                "circuits": CIRCUITS, "training": metadata}
        np.savez_compressed(path, **self.params, mean=self.mean, scale=self.scale,
                            metadata=np.array(json.dumps(info)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as checkpoint:
            info = json.loads(checkpoint["metadata"].item())
            if (info["format"] != "two_head_mlp_v1" or tuple(info["behaviours"]) != BEHAVIOURS
                    or tuple(info["circuits"]) != CIRCUITS):
                raise ValueError("Unrecognized model format or label order")
            model = cls(info["features"], info["hidden_sizes"])
            if set(checkpoint.files) != {*model.params, "mean", "scale", "metadata"}:
                raise ValueError("Checkpoint fields do not match the model")
            for name in (*model.params, "mean", "scale"):
                value = checkpoint[name]
                expected = model.params[name] if name in model.params else getattr(model, name)
                if value.shape != expected.shape or not np.isfinite(value).all():
                    raise ValueError(f"Invalid checkpoint array: {name}")
                if name in model.params:
                    model.params[name] = value.copy()
                else:
                    setattr(model, name, value.copy())
            if (model.scale <= 0).any():
                raise ValueError("Invalid feature scaling")
        return model, info["training"]


class Adam:
    def __init__(self, parameters, learning_rate=.001):
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.first = {name: np.zeros_like(value) for name, value in parameters.items()}
        self.second = {name: np.zeros_like(value) for name, value in parameters.items()}
        self.steps = 0

    def step(self, gradients):
        if set(gradients) != set(self.parameters) or any(
            value.shape != self.parameters[name].shape or not np.isfinite(value).all()
            for name, value in gradients.items()
        ):
            raise ValueError("Invalid gradients; no parameter update was applied")
        self.steps += 1
        for name, parameter in self.parameters.items():
            gradient = gradients[name]
            self.first[name] = .9 * self.first[name] + .1 * gradient
            self.second[name] = .999 * self.second[name] + .001 * gradient ** 2
            mean = self.first[name] / (1 - .9 ** self.steps)
            variance = self.second[name] / (1 - .999 ** self.steps)
            parameter -= self.learning_rate * mean / (np.sqrt(variance) + 1e-8)
