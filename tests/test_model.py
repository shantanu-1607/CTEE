import numpy as np
import pytest

from load_classifier.model import Adam, TwoHeadMLP, cross_entropy, softmax


def test_stable_softmax_and_loss():
    logits = np.array([[10000., -10000., 0.], [-10000., 10000., 0.]])
    np.testing.assert_allclose(softmax(logits).sum(axis=1), 1.)
    loss, gradient = cross_entropy(logits, np.array([1, 0]))
    assert loss == 20000.
    assert np.isfinite(gradient).all()


def test_all_gradients_against_finite_differences():
    rng = np.random.default_rng(17)
    model = TwoHeadMLP(("a", "b", "c"), (4, 3), seed=7)
    x = rng.normal(size=(5, 3))
    behaviour = np.array([0, 1, 2, 0, 1])
    circuit = np.array([0, 2, 4, 6, 3])
    model.fit_scaler(x)
    _, gradients = model.loss_and_gradients(x, behaviour, circuit, l2=.02)
    for name, parameter in model.params.items():
        numerical = np.zeros_like(parameter)
        for index in np.ndindex(parameter.shape):
            value = parameter[index]
            parameter[index] = value + 1e-6
            plus = model.loss_and_gradients(x, behaviour, circuit, l2=.02)[0]
            parameter[index] = value - 1e-6
            minus = model.loss_and_gradients(x, behaviour, circuit, l2=.02)[0]
            parameter[index] = value
            numerical[index] = (plus - minus) / 2e-6
        np.testing.assert_allclose(gradients[name], numerical, atol=1e-7, rtol=1e-5, err_msg=name)


def test_adam_matches_hand_calculation():
    parameters = {"W": np.array([2.])}
    optimizer = Adam(parameters, learning_rate=.1)
    optimizer.step({"W": np.array([4.])})
    expected = 2 - .1 * 4 / (4 + 1e-8)
    np.testing.assert_allclose(parameters["W"], expected)
    optimizer.step({"W": np.array([1.])})
    mean = (.9 * .4 + .1) / (1 - .9 ** 2)
    variance = (.999 * .016 + .001) / (1 - .999 ** 2)
    expected -= .1 * mean / (np.sqrt(variance) + 1e-8)
    np.testing.assert_allclose(parameters["W"], expected)


def test_bad_gradient_does_not_partially_update_model():
    parameters = {"W": np.array([2.]), "b": np.array([3.])}
    optimizer = Adam(parameters)
    with pytest.raises(ValueError):
        optimizer.step({"W": np.array([1.]), "b": np.array([np.nan])})
    assert optimizer.steps == 0
    assert parameters["W"][0] == 2.


def test_scaling_is_training_only_and_checkpoint_round_trip(tmp_path):
    model = TwoHeadMLP(("a", "b"), (4, 3))
    training = np.array([[1., 5.], [3., 5.], [5., 5.]])
    model.fit_scaler(training)
    np.testing.assert_array_equal(model.mean, [3., 5.])
    assert model.scale[1] == 1.
    probabilities = model.predict_proba(np.array([[1000., -1000.]]))
    np.testing.assert_array_equal(model.mean, [3., 5.])
    model.save(tmp_path / "model.npz", {"seed": 42})
    restored, metadata = TwoHeadMLP.load(tmp_path / "model.npz")
    assert metadata == {"seed": 42}
    for head, values in restored.predict_proba(np.array([[1000., -1000.]])).items():
        np.testing.assert_array_equal(values, probabilities[head])


@pytest.mark.parametrize("x", [np.ones((3, 4)), np.array([[np.nan, 1.]])])
def test_wrong_or_nonfinite_inputs_are_rejected(x):
    with pytest.raises(ValueError):
        TwoHeadMLP(("a", "b")).predict_proba(x)
