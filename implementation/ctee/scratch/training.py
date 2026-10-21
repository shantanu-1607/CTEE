"""Mini-batches, task masking, validation, and early stopping without trainers."""
from dataclasses import dataclass
import copy
import math

import numpy as np

from .network import NetworkConfig, TwoHeadMLP
from .optim import Adam, SGD


@dataclass
class TrainingData:
    x: np.ndarray
    behaviour: np.ndarray
    composition: np.ndarray
    behaviour_mask: np.ndarray | None = None
    composition_mask: np.ndarray | None = None
    slopes: np.ndarray | None = None
    slope_mask: np.ndarray | None = None
    priors: dict | None = None

    def __post_init__(self):
        self.x = np.asarray(self.x)
        self.behaviour = np.asarray(self.behaviour)
        self.composition = np.asarray(self.composition)
        n = len(self.x)
        if self.x.ndim != 2 or not n or not np.isfinite(self.x).all():
            raise ValueError("Training inputs must be a nonempty finite matrix")
        for labels, mask_name, classes in [(self.behaviour, "behaviour_mask", 3), (self.composition, "composition_mask", 7)]:
            if labels.shape != (n,) or not np.issubdtype(labels.dtype, np.integer):
                raise ValueError("Training labels must be integer vectors matching the input rows")
            mask = getattr(self, mask_name)
            mask = labels >= 0 if mask is None else np.asarray(mask)
            if mask.shape != (n,) or mask.dtype != np.bool_:
                raise ValueError("Training masks must be boolean vectors")
            if ((labels[mask] < 0) | (labels[mask] >= classes)).any():
                raise ValueError("Observed training label outside its class range")
            setattr(self, mask_name, mask)
        if self.slopes is not None:
            self.slopes = np.asarray(self.slopes)
            self.slope_mask = np.isfinite(self.slopes) if self.slope_mask is None else np.asarray(self.slope_mask)
            if self.slopes.shape != (n,) or self.slope_mask.shape != (n,) or self.slope_mask.dtype != np.bool_:
                raise ValueError("Slope targets and masks must match the input rows")
            if not np.isfinite(self.slopes[self.slope_mask]).all():
                raise ValueError("Observed slope targets must be finite")
        elif self.slope_mask is not None:
            raise ValueError("A slope mask needs slope targets")
        if self.priors is not None:
            if set(self.priors) != {"behaviour", "composition"}:
                raise ValueError("Supply both head priors")
            for name, classes in [("behaviour", 3), ("composition", 7)]:
                p = np.asarray(self.priors[name])
                if p.shape != (n, classes) or not np.isfinite(p).all():
                    raise ValueError("Prior logits must match each head")
                self.priors[name] = p

    def subset(self, ids):
        return TrainingData(self.x[ids], self.behaviour[ids], self.composition[ids],
                            self.behaviour_mask[ids], self.composition_mask[ids],
                            None if self.slopes is None else self.slopes[ids],
                            None if self.slope_mask is None else self.slope_mask[ids],
                            None if self.priors is None else {n: p[ids] for n, p in self.priors.items()})


@dataclass(frozen=True)
class TrainingConfig:
    network: NetworkConfig
    optimizer: str = "adam"
    learning_rate: float = 0.001
    batch_size: int = 128
    epochs: int = 100
    patience: int = 16
    min_delta: float = 1e-6
    schedule: str = "constant"
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    momentum: float = 0.9
    nesterov: bool = False
    weight_decay: float = 0.0001
    clip_norm: float | None = 5.0
    behaviour_weight: float = 1.0
    composition_weight: float = 1.0
    label_smoothing: float = 0.0
    behaviour_class_weights: tuple | None = None
    composition_class_weights: tuple | None = None
    l2: float = 0.0
    physics_weight: float = 0.0
    huber_delta: float = 1.0
    ema_decay: float | None = None

    def __post_init__(self):
        if self.ema_decay is not None and (not np.isfinite(self.ema_decay) or not 0 < self.ema_decay < 1):
            raise ValueError("EMA decay must be between zero and one, or None")
        if self.optimizer not in {"adam", "sgd"} or self.schedule not in {"constant", "cosine"}:
            raise ValueError("Unknown optimiser or learning-rate schedule")
        if any(not isinstance(v, int) or v < 1 for v in [self.batch_size, self.epochs, self.patience]):
            raise ValueError("Batch size, epochs, and patience must be positive integers")
        if not np.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError("min_delta must be finite and nonnegative")
        if not np.isfinite([self.behaviour_weight, self.composition_weight, self.l2, self.physics_weight]).all():
            raise ValueError("Loss settings must be finite")
        if min(self.behaviour_weight, self.composition_weight, self.l2, self.physics_weight) < 0:
            raise ValueError("Loss weights must be nonnegative")
        if self.behaviour_weight + self.composition_weight <= 0 or not 0 <= self.label_smoothing < 1:
            raise ValueError("Enable a supervised task and choose valid label smoothing")
        if not np.isfinite(self.huber_delta) or self.huber_delta <= 0:
            raise ValueError("Huber delta must be positive")


def update_ema(averages, parameters, decay):
    """Exponential averaging of actual learned parameters, not output predictions."""
    if not np.isfinite(decay) or not 0 < decay < 1 or set(averages)!=set(parameters):
        raise ValueError("Invalid EMA decay or parameter keys")
    for name,value in parameters.items():
        if averages[name].shape!=value.shape or not np.isfinite(value).all() or not np.isfinite(averages[name]).all():
            raise ValueError("Invalid EMA parameter arrays")
    for name,value in parameters.items():averages[name][...] = decay*averages[name]+(1-decay)*value


def _loss(model, data, config, training):
    return model.loss_and_gradients(
        data.x, data.behaviour, data.composition,
        behaviour_mask=data.behaviour_mask, composition_mask=data.composition_mask,
        behaviour_weight=config.behaviour_weight, composition_weight=config.composition_weight,
        smoothing=config.label_smoothing, behaviour_class_weights=config.behaviour_class_weights,
        composition_class_weights=config.composition_class_weights, l2=config.l2,
        physics_weight=config.physics_weight, slopes=data.slopes, slope_mask=data.slope_mask,
        huber_delta=config.huber_delta, training=training, priors=data.priors)


def fit(training_data, validation_data, config):
    """Fit only training_data; select weights only by validation loss.

    This API never takes a test split. The caller owns parent-group splitting.
    Inference checkpoints save weights/scaling, not resumable optimiser state.
    """
    if training_data.x.shape[1] != config.network.input_dim or validation_data.x.shape[1] != config.network.input_dim:
        raise ValueError("Feature counts must match the network")
    for data in [training_data, validation_data]:
        if not ((config.behaviour_weight > 0 and data.behaviour_mask.any()) or
                (config.composition_weight > 0 and data.composition_mask.any())):
            raise ValueError("Each dataset needs observed labels for an enabled task")
        if config.physics_weight and data.slopes is None:
            raise ValueError("Physics training requires measured slope targets")
    model = TwoHeadMLP(config.network)
    model.fit_scaler(training_data.x)
    initial_params = copy.deepcopy(model.params)
    averaged_params = copy.deepcopy(model.params) if config.ema_decay is not None else None
    common = dict(learning_rate=config.learning_rate, weight_decay=config.weight_decay, clip_norm=config.clip_norm)
    optimizer = (Adam(model.params, beta1=config.beta1, beta2=config.beta2, epsilon=config.epsilon, **common)
                 if config.optimizer == "adam" else SGD(model.params, momentum=config.momentum, nesterov=config.nesterov, **common))
    rng = np.random.default_rng(config.network.seed + 1)
    best = float("inf")
    best_params = None
    patience = 0
    history = []
    best_epoch = 0
    best_weight_source = "raw"
    for epoch in range(config.epochs):
        if config.schedule == "cosine":
            fraction = epoch / max(config.epochs - 1, 1)
            optimizer.learning_rate = config.learning_rate * (0.01 + 0.99 * (1 + math.cos(math.pi * fraction)) / 2)
        order = rng.permutation(len(training_data.x))
        total = 0.0
        gradient_norms = []
        epoch_start = copy.deepcopy(model.params)
        for start in range(0, len(order), config.batch_size):
            batch = training_data.subset(order[start:start + config.batch_size])
            details, grads = _loss(model, batch, config, True)
            if not np.isfinite(details["loss"]):
                raise ValueError("Non-finite training loss")
            update = optimizer.step(grads)
            gradient_norms.append(update["unclipped_gradient_norm"])
            if averaged_params is not None:update_ema(averaged_params,model.params,config.ema_decay)
            total += details["loss"] * len(batch.x)
        validation, _ = _loss(model, validation_data, config, False)
        raw_loss = validation["loss"];averaged_loss = None;candidate_params=model.params;weight_source="raw"
        if averaged_params is not None:
            raw_params=model.params
            try:
                model.params=averaged_params
                averaged_validation,_=_loss(model,validation_data,config,False)
                averaged_loss=averaged_validation['loss']
            finally:model.params=raw_params
            if not np.isfinite(averaged_loss):raise ValueError("Non-finite averaged validation loss")
            if averaged_loss<raw_loss:candidate_params=averaged_params;weight_source="ema"
        loss = min(raw_loss,averaged_loss) if averaged_loss is not None else raw_loss
        if not np.isfinite(loss):
            raise ValueError("Non-finite validation loss")
        history.append({"epoch": epoch + 1, "training_loss": total / len(training_data.x),
                        "validation_loss": loss, "raw_validation_loss": raw_loss,
                        "averaged_validation_loss": averaged_loss, "weight_source": weight_source,
                        "learning_rate": optimizer.learning_rate,
                        "mean_gradient_norm": float(np.mean(gradient_norms)),
                        "parameter_update_l2": float(np.sqrt(sum(np.sum((model.params[k]-v)**2) for k,v in epoch_start.items())))})
        if loss < best - config.min_delta:
            best = loss
            best_params = copy.deepcopy(candidate_params)
            best_weight_source = weight_source
            best_epoch = epoch + 1
            patience = 0
        else:
            patience += 1
        if patience >= config.patience:
            break
    model.params = best_params
    return model, {"history": history, "best_epoch": best_epoch, "best_validation_loss": best,
                   "optimizer_steps": optimizer.steps, "selection": "validation loss only",
                   "weight_source": best_weight_source, "ema_decay":config.ema_decay,
                   "parameter_learning": {k:{"shape":list(v.shape), "count":int(v.size),
                       "initial_l2":float(np.sqrt(np.sum(initial_params[k]**2))),
                       "learned_l2":float(np.sqrt(np.sum(v**2))),
                       "change_l2":float(np.sqrt(np.sum((v-initial_params[k])**2))),
                       "changed_values":int(np.sum(abs(v-initial_params[k])>1e-12))} for k,v in model.params.items()}}
