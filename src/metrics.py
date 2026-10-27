"""Rating metrics with explicit input validation."""

import numpy as np


def _errors(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.ndim != 1 or actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("Metrics need equal-length, nonempty one-dimensional arrays")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrics require finite values")
    return actual - predicted


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Return root mean squared error."""
    return float(np.sqrt(np.mean(_errors(actual, predicted) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Return mean absolute error."""
    return float(np.mean(np.abs(_errors(actual, predicted))))
