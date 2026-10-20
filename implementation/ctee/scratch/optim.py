"""Explicit AdamW-style and momentum-SGD updates with global norm clipping."""
import numpy as np


def _validated_gradients(params, gradients, clip_norm):
    if set(params) != set(gradients):
        raise ValueError("One gradient is required for every model parameter")
    for name, gradient in gradients.items():
        if gradient.shape != params[name].shape or not np.isfinite(gradient).all() or not np.isfinite(params[name]).all():
            raise ValueError("Gradient shape mismatch or non-finite gradient")
    norm = float(np.sqrt(sum(float(np.sum(g.astype(np.float64)**2)) for g in gradients.values())))
    if not np.isfinite(norm):
        raise ValueError("Gradient norm overflow")
    if clip_norm is not None and (not np.isfinite(clip_norm) or clip_norm <= 0):
        raise ValueError("clip_norm must be positive or None")
    factor = min(1.0, clip_norm / max(norm, 1e-30)) if clip_norm is not None else 1.0
    return {name: value * factor for name, value in gradients.items()}, norm


def _settings(learning_rate, weight_decay):
    if not np.isfinite([learning_rate, weight_decay]).all() or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("Learning rate must be positive and weight decay nonnegative")


class Adam:
    def __init__(self, params, learning_rate=0.001, beta1=0.9, beta2=0.999,
                 epsilon=1e-8, weight_decay=0.0, clip_norm=5.0):
        _settings(learning_rate, weight_decay)
        if not (0 <= beta1 < 1 and 0 <= beta2 < 1 and np.isfinite(epsilon) and epsilon > 0):
            raise ValueError("Invalid Adam moment or epsilon")
        self.params = params
        self.learning_rate = learning_rate
        self.beta1, self.beta2, self.epsilon = beta1, beta2, epsilon
        self.weight_decay, self.clip_norm = weight_decay, clip_norm
        self.first = {n: np.zeros_like(p) for n, p in params.items()}
        self.second = {n: np.zeros_like(p) for n, p in params.items()}
        self.steps = 0

    def step(self, gradients):
        _settings(self.learning_rate, self.weight_decay)
        grads, norm = _validated_gradients(self.params, gradients, self.clip_norm)
        self.steps += 1
        for name, p in self.params.items():
            g = grads[name]
            self.first[name][...] = self.beta1 * self.first[name] + (1 - self.beta1) * g
            self.second[name][...] = self.beta2 * self.second[name] + (1 - self.beta2) * g**2
            mhat = self.first[name] / (1 - self.beta1**self.steps)
            vhat = self.second[name] / (1 - self.beta2**self.steps)
            decay = self.weight_decay * p if name.startswith("W") else 0.0
            p -= self.learning_rate * (mhat / (np.sqrt(vhat) + self.epsilon) + decay)
        return {"unclipped_gradient_norm": norm, "step": self.steps}


class SGD:
    def __init__(self, params, learning_rate=0.01, momentum=0.0, nesterov=False,
                 weight_decay=0.0, clip_norm=5.0):
        _settings(learning_rate, weight_decay)
        if not np.isfinite(momentum) or not 0 <= momentum < 1 or (nesterov and momentum == 0):
            raise ValueError("Invalid momentum/Nesterov setting")
        self.params, self.learning_rate = params, learning_rate
        self.momentum, self.nesterov = momentum, nesterov
        self.weight_decay, self.clip_norm = weight_decay, clip_norm
        self.velocity = {n: np.zeros_like(p) for n, p in params.items()}
        self.steps = 0

    def step(self, gradients):
        _settings(self.learning_rate, self.weight_decay)
        grads, norm = _validated_gradients(self.params, gradients, self.clip_norm)
        self.steps += 1
        for name, p in self.params.items():
            g = grads[name]
            self.velocity[name][...] = self.momentum * self.velocity[name] + g
            direction = g + self.momentum * self.velocity[name] if self.nesterov else self.velocity[name]
            decay = self.weight_decay * p if name.startswith("W") else 0.0
            p -= self.learning_rate * (direction + decay)
        return {"unclipped_gradient_norm": norm, "step": self.steps}
