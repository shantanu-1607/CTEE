from dataclasses import replace

import numpy as np

from load_classifier.train import TrainingConfig, fit


def toy_data(n, seed):
    rng = np.random.default_rng(seed)
    behaviour = np.arange(n) % 3
    circuit = (np.arange(n) // 3) % 7
    x = np.column_stack((np.eye(3)[behaviour], np.eye(7)[circuit]))
    return x + rng.normal(0, .01, x.shape), behaviour, circuit


def test_training_learns_both_outputs():
    training, validation = toy_data(420, 3), toy_data(210, 4)
    features = tuple(f"x{i}" for i in range(10))
    config = TrainingConfig(hidden_sizes=(24, 16), learning_rate=.01, max_epochs=60, l2=0.)
    model, history, record = fit(training, validation, config, features, verbose=False)
    probabilities = model.predict_proba(validation[0])
    assert np.mean(probabilities["behaviour"].argmax(axis=1) == validation[1]) > .98
    assert np.mean(probabilities["circuit"].argmax(axis=1) == validation[2]) > .98
    assert record["best_validation_loss"] < history.iloc[0]["validation_loss"] * .1


def test_early_stopping_restores_best_weights():
    training, validation = toy_data(84, 1), toy_data(42, 2)
    features = tuple(f"x{i}" for i in range(10))
    config = TrainingConfig(hidden_sizes=(8,), max_epochs=10, patience=2, min_delta=100., batch_size=84)
    model, history, record = fit(training, validation, config, features, verbose=False)
    first, _, _ = fit(training, validation, replace(config, max_epochs=1), features, verbose=False)
    assert record["best_epoch"] == 1
    assert len(history) == 3
    for name in model.params:
        np.testing.assert_array_equal(model.params[name], first.params[name])
