"""Mini-batch training; checkpoints are selected using validation loss only."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .data import FEATURES, arrays, file_hash
from .metrics import score
from .model import Adam, TwoHeadMLP


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    hidden_sizes: tuple = (64, 32)
    learning_rate: float = .001
    batch_size: int = 128
    max_epochs: int = 250
    patience: int = 25
    min_delta: float = .00001
    l2: float = .0001

    def __post_init__(self):
        if (not self.hidden_sizes or any(type(n) is not int or n < 1 for n in self.hidden_sizes)
                or any(type(n) is not int or n < 1 for n in (self.batch_size, self.max_epochs, self.patience))
                or type(self.seed) is not int or self.seed < 0):
            raise ValueError("Invalid network sizes, epoch counts or seed")
        if (not np.isfinite([self.learning_rate, self.min_delta, self.l2]).all()
                or self.learning_rate <= 0 or min(self.min_delta, self.l2) < 0):
            raise ValueError("Invalid learning rate, stopping threshold or L2 penalty")

    @classmethod
    def read(cls, path):
        values = json.loads(Path(path).read_text())
        values["hidden_sizes"] = tuple(values["hidden_sizes"])
        return cls(**values)


def fit(training, validation, config, features=FEATURES, verbose=True):
    """Accept only train/validation arrays; the test partition is never passed in."""
    x_train, b_train, c_train = training
    x_val, b_val, c_val = validation
    model = TwoHeadMLP(features, config.hidden_sizes, config.seed)
    model.fit_scaler(x_train)
    optimizer = Adam(model.params, config.learning_rate)
    rng = np.random.default_rng(config.seed + 1)
    history = []
    best_loss = np.inf
    best_epoch = 0
    best_parameters = None
    stale_epochs = 0
    for epoch in range(1, config.max_epochs + 1):
        order = rng.permutation(len(x_train))
        for start in range(0, len(order), config.batch_size):
            batch = order[start:start + config.batch_size]
            loss, gradients = model.loss_and_gradients(
                x_train[batch], b_train[batch], c_train[batch], config.l2)
            if not np.isfinite(loss):
                raise ValueError("Non-finite training loss")
            optimizer.step(gradients)
        training_loss = model.loss(x_train, b_train, c_train)
        validation_loss = model.loss(x_val, b_val, c_val)
        if not np.isfinite([training_loss, validation_loss]).all():
            raise ValueError("Non-finite epoch loss")
        validation_scores = score(b_val, c_val, model.predict_proba(x_val))
        history.append({"epoch": epoch, "training_loss": training_loss,
                        "validation_loss": validation_loss,
                        "validation_behaviour_accuracy": validation_scores["behaviour"]["accuracy"],
                        "validation_circuit_accuracy": validation_scores["circuit"]["accuracy"],
                        "validation_joint_accuracy": validation_scores["joint_accuracy"]})
        if validation_loss < best_loss - config.min_delta:
            best_loss, best_epoch = validation_loss, epoch
            best_parameters = {name: value.copy() for name, value in model.params.items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(f"Epoch {epoch:3d}: train loss {training_loss:.4f}, validation loss "
                  f"{validation_loss:.4f}, joint {validation_scores['joint_accuracy']:.2%}", flush=True)
        if stale_epochs >= config.patience:
            break
    model.params = best_parameters
    return model, pd.DataFrame(history), {"best_epoch": best_epoch,
        "epochs_completed": len(history), "optimizer_updates": optimizer.steps,
        "best_validation_loss": float(best_loss),
        "selection": "Lowest validation sum of behaviour and circuit cross-entropy, min_delta applied"}


def train_run(data, config, output, workbook, split_path, features=FEATURES):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Run folder is not empty; choose a new --run for another experiment")
    started = time.perf_counter()
    model, history, metadata = fit(arrays(data[data["split"] == "train"], features),
                                   arrays(data[data["split"] == "validation"], features), config, features)
    metadata.update({"config": asdict(config), "workbook_sha256": file_hash(workbook),
        "splits_sha256": file_hash(split_path), "features": list(features),
        "training_rows": int((data["split"] == "train").sum()),
        "validation_rows": int((data["split"] == "validation").sum()),
        "parameter_count": sum(value.size for value in model.params.values()),
        "training_seconds": time.perf_counter() - started,
        "numpy_version": np.__version__})
    history.to_csv(output / "history.csv", index=False)
    model.save(output / "model.npz", metadata)
    metadata["checkpoint_sha256"] = file_hash(output / "model.npz")
    (output / "training.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved epoch {metadata['best_epoch']} to {output / 'model.npz'}", flush=True)
    return metadata
