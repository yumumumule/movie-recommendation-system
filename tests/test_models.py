"""Baseline arithmetic, finite-difference gradients, stopping and persistence."""

import numpy as np
import pandas as pd
import pytest

from src.metrics import rmse
from src.models import BiasCF, GlobalMean, ItemMean, MatrixFactorization, UserMean, load_model


def test_baseline_arithmetic(ratings):
    np.testing.assert_allclose(GlobalMean().fit(ratings).predict([10, 999], [5, 999]), [3, 3])
    np.testing.assert_allclose(
        UserMean().fit(ratings).predict([10, 30, 999], [5, 5, 5]), [4, 2, 3]
    )
    np.testing.assert_allclose(ItemMean().fit(ratings).predict([10, 999], [20, 999]), [3, 3])
    unbalanced = ratings.iloc[[0, 1, 2, 3]].copy()
    assert ItemMean().fit(unbalanced).predict([999], [20])[0] == 4


def test_bias_coordinate_solution(ratings):
    model = BiasCF(regularization=0, epochs=50).fit(ratings)
    assert model.user_bias[0] == pytest.approx(1)
    assert model.user_bias[1] == pytest.approx(-1)
    np.testing.assert_allclose(model.item_bias, 0, atol=1e-10)


@pytest.mark.parametrize(
    "factory", [GlobalMean, UserMean, ItemMean, BiasCF, lambda: MatrixFactorization(epochs=3)]
)
def test_cold_start_finite_and_serialization(factory, ratings, tmp_path):
    model = factory().fit(ratings)
    users, items = np.array([10, 999, 10, 999]), np.array([5, 5, 999, 999])
    prediction = model.predict(users, items)
    assert np.isfinite(prediction).all()
    assert prediction[-1] == model.global_mean
    assert prediction[1] == pytest.approx(np.clip(model.global_mean + model.item_bias[0], 0.5, 5))
    assert prediction[2] == pytest.approx(np.clip(model.global_mean + model.user_bias[0], 0.5, 5))
    path = tmp_path / "model.npz"
    model.save(path)
    restored = load_model(path)
    np.testing.assert_array_equal(restored.predict(users, items), prediction)
    np.testing.assert_array_equal(restored.users.ids, [10, 30])
    np.testing.assert_array_equal(restored.seen_items, model.seen_items)
    assert restored.get_params() == model.get_params()


def test_sgd_matches_numerical_gradient_with_repeated_ids(ratings):
    model = MatrixFactorization(
        n_factors=2, learning_rate=0.001, regularization=0.2, epochs=1
    ).fit(ratings)
    u, i, r = np.array([0, 0, 1]), np.array([0, 1, 1]), np.array([5.0, 4.0, 2.0])
    arrays = [model.user_bias, model.item_bias, model.user_factors, model.item_factors]
    originals = [array.copy() for array in arrays]

    def objective():
        bu, bi, p, q = arrays
        error = r - model.global_mean - bu[u] - bi[i] - np.sum(p[u] * q[i], axis=1)
        penalty = bu[u] ** 2 + bi[i] ** 2 + np.sum(p[u] ** 2 + q[i] ** 2, axis=1)
        return 0.5 * np.sum(error**2 + model.regularization * penalty)

    gradients = []
    epsilon = 1e-6
    for array in arrays:
        gradient = np.zeros_like(array)
        for index in np.ndindex(array.shape):
            old = array[index]
            array[index] = old + epsilon
            high = objective()
            array[index] = old - epsilon
            low = objective()
            array[index] = old
            gradient[index] = (high - low) / (2 * epsilon)
        gradients.append(gradient)
    model._sgd_batch(u, i, r)
    updated = [model.user_bias, model.item_bias, model.user_factors, model.item_factors]
    for actual, before, gradient in zip(updated, originals, gradients):
        np.testing.assert_allclose(actual, before - model.learning_rate * gradient, atol=1e-10)


def test_factorization_learns_interactions():
    rows = [(u, i, 5.0 if (u % 2) == (i % 2) else 1.0) for u in range(1, 9) for i in range(1, 9)]
    frame = pd.DataFrame(rows, columns=["userId", "movieId", "rating"])
    model = MatrixFactorization(
        n_factors=3, regularization=0.001, learning_rate=0.01, epochs=150, batch_size=8
    ).fit(frame)
    prediction = model.predict(frame.userId.to_numpy(), frame.movieId.to_numpy())
    assert rmse(frame.rating.to_numpy(), prediction) < 0.15
    assert model.history[-1]["train_loss"] < model.history[0]["train_loss"]


def test_early_stopping_restores_checkpoint(ratings):
    validation = pd.DataFrame({"userId": [999], "movieId": [999], "rating": [3.0]})
    model = MatrixFactorization(epochs=20, patience=2).fit(ratings, validation)
    first_epoch = MatrixFactorization(epochs=1).fit(ratings)
    assert len(model.history) == 3
    assert model.best_epoch == 1
    np.testing.assert_array_equal(model.user_factors, first_epoch.user_factors)


def test_seed_determinism_and_validation_overlap(ratings):
    first = MatrixFactorization(epochs=4).fit(ratings)
    second = MatrixFactorization(epochs=4).fit(ratings)
    np.testing.assert_array_equal(first.user_factors, second.user_factors)
    with pytest.raises(ValueError, match="overlap"):
        MatrixFactorization().fit(ratings, ratings.iloc[:1])


@pytest.mark.parametrize(
    "params",
    [{"n_factors": 0}, {"learning_rate": -1}, {"regularization": -1}, {"learning_rate": np.nan}],
)
def test_invalid_hyperparameters(params):
    with pytest.raises(ValueError):
        MatrixFactorization(**params)


def test_unfitted_and_missing_model(tmp_path):
    with pytest.raises(ValueError, match="not fitted"):
        GlobalMean().predict([1], [1])
    with pytest.raises(FileNotFoundError, match="src.train"):
        load_model(tmp_path / "missing.npz")
