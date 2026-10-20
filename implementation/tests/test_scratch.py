"""Independent arithmetic checks; these tests do not use a library ML oracle."""
import ast
import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ctee.scratch import Adam, NetworkConfig, SGD, TrainingConfig, TrainingData, TwoHeadMLP, fit, softmax
from ctee.scratch.network import cross_entropy, slope_consistency


def test_stable_softmax_and_cross_entropy():
    z = np.array([[10000., -10000., 0.], [-10000., 10000., 0.]])
    p = softmax(z)
    np.testing.assert_allclose(p.sum(1), 1.)
    loss, grad = cross_entropy(z, np.array([1, 0]))
    assert np.isfinite(loss) and np.isfinite(grad).all()
    assert loss == 20000.
    np.testing.assert_allclose(grad.sum(1), 0., atol=1e-15)


def test_weighted_smoothed_cross_entropy_matches_hand_calculation():
    z = np.array([[1., 2., -1.], [.4, .5, -.2]])
    y = np.array([0, 1])
    weights = np.array([2., 3., 1.])
    p = softmax(z)
    target = np.full((2, 3), .2 / 3)
    target[np.arange(2), y] += .8
    expected = -np.sum(weights[y, None] * target * np.log(p)) / 5
    loss, grad = cross_entropy(z, y, smoothing=.2, class_weights=weights)
    assert abs(loss - expected) < 1e-12
    np.testing.assert_allclose(grad, (p - target) * weights[y, None] / 5)


def test_unobserved_labels_do_not_contribute():
    z = np.array([[1., 0., -1.], [.2, .3, .1]])
    loss, gradient = cross_entropy(z, np.array([-1, 1]), np.array([False, True]))
    single, _ = cross_entropy(z[1:], np.array([1]))
    assert loss == single
    assert np.all(gradient[0] == 0)
    empty, grad = cross_entropy(z, np.array([-1, -1]), np.zeros(2, dtype=bool))
    assert empty == 0 and np.all(grad == 0)


@pytest.mark.parametrize("activation", ["relu", "leaky_relu", "tanh"])
def test_full_backpropagation_against_finite_differences(activation):
    rng = np.random.default_rng(52)
    model = TwoHeadMLP(NetworkConfig(4, (5, 3), activation=activation, dropout=0., seed=43))
    x = rng.normal(size=(5, 4))
    model.fit_scaler(x * 1.2 + .3)
    yb = np.array([0, 1, 2, -1, 0])
    yc = np.array([6, 2, 1, 0, 3])
    settings = dict(training=False, behaviour_mask=yb >= 0,
                    behaviour_weight=.7, composition_weight=1.3, smoothing=.1,
                    behaviour_class_weights=(1., 2., 1.5), l2=.017,
                    physics_weight=.2, slopes=np.array([-.4, -.8, .9, .1, .6]),
                    slope_mask=np.array([True, True, False, True, True]), huber_delta=.3,
                    priors={"behaviour":rng.normal(size=(5,3)), "composition":rng.normal(size=(5,7))})
    model = TwoHeadMLP(replace(model.config, prior_strength=.25))
    model.fit_scaler(x * 1.2 + .3)
    _, analytical = model.loss_and_gradients(x, yb, yc, **settings)
    step = 1e-6
    for name, parameter in model.params.items():
        numerical = np.zeros_like(parameter)
        for ix in np.ndindex(parameter.shape):
            original = parameter[ix]
            parameter[ix] = original + step
            plus = model.loss_and_gradients(x, yb, yc, **settings)[0]["loss"]
            parameter[ix] = original - step
            minus = model.loss_and_gradients(x, yb, yc, **settings)[0]["loss"]
            parameter[ix] = original
            numerical[ix] = (plus - minus) / (2 * step)
        np.testing.assert_allclose(analytical[name], numerical, atol=2e-7, rtol=2e-5, err_msg=name)


def test_shared_encoder_receives_sum_of_both_head_gradients():
    model = TwoHeadMLP(NetworkConfig(3, (4, 3), activation="tanh", dropout=0.))
    x = np.random.default_rng(13).normal(size=(4, 3))
    yb = np.array([0, 1, 2, 0]); yc = np.array([6, 3, 1, 4])
    _, both = model.loss_and_gradients(x, yb, yc, training=False)
    _, only_b = model.loss_and_gradients(x, yb, yc, composition_weight=0., training=False)
    _, only_c = model.loss_and_gradients(x, yb, yc, behaviour_weight=0., training=False)
    for name in model.params:
        np.testing.assert_allclose(both[name], only_b[name] + only_c[name], atol=1e-14)


def test_inverted_dropout_is_disabled_for_inference():
    model = TwoHeadMLP(NetworkConfig(3, (5, 4), activation="tanh", dropout=.5))
    x = np.random.default_rng(8).normal(size=(30, 3))
    a = model.forward(x)
    b = model.forward(x)
    np.testing.assert_array_equal(a["composition"], b["composition"])
    trained, _ = model._forward(x, True, None)
    assert not np.allclose(trained["composition"], a["composition"])


def test_dropout_backpropagation_with_a_fixed_mask():
    model = TwoHeadMLP(NetworkConfig(3, (4,), activation="tanh", dropout=.25, seed=9))
    x = np.random.default_rng(3).normal(size=(4, 3))
    yb = np.array([0, 1, 2, 1]); yc = np.array([0, 2, 3, 6])
    state = copy.deepcopy(model.rng.bit_generator.state)
    _, analytical = model.loss_and_gradients(x, yb, yc)
    for name, parameter in model.params.items():
        numerical = np.zeros_like(parameter)
        for ix in np.ndindex(parameter.shape):
            original = parameter[ix]
            parameter[ix] = original + 1e-6
            model.rng.bit_generator.state = copy.deepcopy(state)
            plus = model.loss_and_gradients(x, yb, yc)[0]["loss"]
            parameter[ix] = original - 1e-6
            model.rng.bit_generator.state = copy.deepcopy(state)
            minus = model.loss_and_gradients(x, yb, yc)[0]["loss"]
            parameter[ix] = original
            numerical[ix] = (plus - minus) / 2e-6
        np.testing.assert_allclose(analytical[name], numerical, atol=2e-7, rtol=2e-5)


def test_adam_bias_correction_and_weight_decay_by_hand():
    params = {"W0":np.array([2.]), "b0":np.array([3.])}
    opt = Adam(params, learning_rate=.1, beta1=.8, beta2=.9, epsilon=1e-8, weight_decay=.2, clip_norm=None)
    first = {"W0":np.array([4.]), "b0":np.array([-2.])}
    opt.step(first)
    expected_w = 2 - .1 * (4 / (4 + 1e-8) + .2 * 2)
    expected_b = 3 - .1 * (-2 / (2 + 1e-8))
    np.testing.assert_allclose(params["W0"], expected_w)
    np.testing.assert_allclose(params["b0"], expected_b)
    opt.step({"W0":np.array([1.]), "b0":np.array([1.])})
    mhat = (.8 * (.2 * 4) + .2 * 1) / (1 - .8**2)
    vhat = (.9 * (.1 * 4**2) + .1 * 1) / (1 - .9**2)
    expected_w -= .1 * (mhat / (np.sqrt(vhat) + 1e-8) + .2 * expected_w)
    np.testing.assert_allclose(params["W0"], expected_w)


def test_sgd_momentum_and_nesterov_by_hand():
    for nesterov in [False, True]:
        p = {"W0":np.array([2.])}
        opt = SGD(p, learning_rate=.1, momentum=.5, nesterov=nesterov, clip_norm=None)
        opt.step({"W0":np.array([3.])})
        first = 4.5 if nesterov else 3.
        np.testing.assert_allclose(p["W0"], 2 - .1 * first)
        opt.step({"W0":np.array([1.])})
        second = 2.25 if nesterov else 2.5
        np.testing.assert_allclose(p["W0"], 2 - .1 * (first + second))


def test_global_gradient_clipping_does_not_modify_supplied_gradients():
    p = {"W0":np.array([0., 0.])}
    g = {"W0":np.array([3., 4.])}
    opt = SGD(p, learning_rate=1., clip_norm=1.)
    details = opt.step(g)
    assert details["unclipped_gradient_norm"] == 5.
    np.testing.assert_allclose(p["W0"], [-.6, -.8])
    np.testing.assert_array_equal(g["W0"], [3., 4.])


def test_invalid_gradients_are_rejected_before_any_update():
    p = {"W0":np.array([2.]), "b0":np.array([3.])}
    opt = Adam(p)
    with pytest.raises(ValueError):
        opt.step({"W0":np.array([1.]), "b0":np.array([np.nan])})
    assert opt.steps == 0
    assert p["W0"][0] == 2 and p["b0"][0] == 3


def test_npz_checkpoint_roundtrip_and_train_only_scaler(tmp_path):
    model = TwoHeadMLP(NetworkConfig(3, (5,), dropout=.2))
    train = np.array([[1., 2., 3.], [3., 2., 5.], [5., 2., 7.]])
    model.fit_scaler(train)
    before_mean = model.mean.copy()
    model.forward(np.array([[100., 200., 300.]]))
    np.testing.assert_array_equal(model.mean, before_mean)
    assert model.scale[1] == model.config.scale_floor
    path = tmp_path / "manual.npz"
    model.save(path)
    restored = TwoHeadMLP.load(path)
    for head, p in model.predict_proba(train).items():
        np.testing.assert_array_equal(p, restored.predict_proba(train)[head])


def _toy_data(n, seed):
    rng = np.random.default_rng(seed)
    b = np.arange(n) % 3
    c = (np.arange(n) // 3) % 7
    x = np.concatenate([np.eye(3)[b], np.eye(7)[c]], axis=1) + rng.normal(0, .02, (n, 10))
    return TrainingData(x, b, c)


def test_manual_training_learns_both_tasks_and_selects_by_validation():
    train, validation = _toy_data(420, 10), _toy_data(210, 20)
    config = TrainingConfig(NetworkConfig(10, (24, 16), dropout=.05, seed=2),
                            learning_rate=.02, epochs=100, patience=16, weight_decay=0., batch_size=64)
    model, report = fit(train, validation, config)
    probabilities = model.predict_proba(validation.x)
    assert np.mean(probabilities["behaviour"].argmax(1) == validation.behaviour) > .98
    assert np.mean(probabilities["composition"].argmax(1) == validation.composition) > .98
    assert report["best_validation_loss"] < .2 * report["history"][0]["validation_loss"]
    assert report["selection"] == "validation loss only"
    assert report["best_epoch"] >= 1


def test_early_stopping_restores_best_validation_weights():
    train, validation = _toy_data(84, 1), _toy_data(42, 2)
    config = TrainingConfig(NetworkConfig(10, (8,), dropout=0., seed=3),
                            epochs=12, patience=2, min_delta=100., batch_size=84, weight_decay=0.)
    restored, report = fit(train, validation, config)
    one_epoch, _ = fit(train, validation, replace(config, epochs=1))
    assert report["best_epoch"] == 1 and len(report["history"]) == 3
    for name in restored.params:
        np.testing.assert_array_equal(restored.params[name], one_epoch.params[name])


def test_missing_labels_and_slopes_have_valid_masks():
    data = TrainingData(np.ones((3, 2)), np.array([0, -1, 2]), np.array([-1, 3, 6]),
                        slopes=np.array([0., np.nan, 1.]))
    np.testing.assert_array_equal(data.behaviour_mask, [True, False, True])
    np.testing.assert_array_equal(data.composition_mask, [False, True, True])
    np.testing.assert_array_equal(data.slope_mask, [True, False, True])
    with pytest.raises(ValueError):
        slope_consistency(np.ones((3, 3)), np.array([0., np.nan, 1.]))


def test_scratch_core_imports_only_numpy_and_standard_python():
    folder = Path(__file__).resolve().parents[1] / "ctee/scratch"
    allowed = {"dataclasses", "copy", "json", "math", "pathlib", "numpy"}
    for source in folder.glob("*.py"):
        tree = ast.parse(source.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(alias.name.split('.')[0] in allowed for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and not node.level:
                assert node.module.split('.')[0] in allowed


@pytest.mark.parametrize("settings", [{"dropout":1.}, {"activation":"unknown"}, {"hidden_widths":()}, {"prior_strength":-1.}])
def test_invalid_network_settings_fail(settings):
    with pytest.raises(ValueError):
        NetworkConfig(3, **settings)
