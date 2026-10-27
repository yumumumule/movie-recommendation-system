"""Seeded, disjoint holdout splitting and train-only ID encoding."""

import numpy as np
import pandas as pd

from src.data_loader import validate_ratings


def split_ratings(
    ratings: pd.DataFrame,
    seed: int = 42,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Randomly hold out interactions globally, without enforcing warm-start coverage.

    This measures missing-rating interpolation, not prediction of future behavior.
    IDs and timestamps are never used as features or to learn from held-out labels.
    """
    validate_ratings(ratings)
    if not (
        0 < validation_fraction < 1
        and 0 < test_fraction < 1
        and validation_fraction + test_fraction < 1
    ):
        raise ValueError("Split fractions must be positive and sum to less than one")
    n_validation = int(len(ratings) * validation_fraction)
    n_test = int(len(ratings) * test_fraction)
    if min(n_validation, n_test, len(ratings) - n_validation - n_test) < 1:
        raise ValueError("Not enough interactions for three nonempty splits")
    order = np.random.default_rng(seed).permutation(len(ratings))
    train_end = len(ratings) - n_validation - n_test
    val_end = train_end + n_validation
    return tuple(
        ratings.iloc[index].reset_index(drop=True)
        for index in (order[:train_end], order[train_end:val_end], order[val_end:])
    )


class IdEncoder:
    """Map sorted training IDs to dense indices; map unseen IDs to -1."""

    def __init__(self, ids: np.ndarray):
        self.ids = np.unique(np.asarray(ids, dtype=np.int64))

    def transform(self, ids: np.ndarray) -> np.ndarray:
        """Encode a one-dimensional sequence without expanding the vocabulary."""
        values = np.asarray(ids)
        if values.ndim != 1:
            raise ValueError("IDs must be one-dimensional")
        indices = np.searchsorted(self.ids, values)
        safe = np.minimum(indices, len(self.ids) - 1)
        return np.where((indices < len(self.ids)) & (self.ids[safe] == values), indices, -1)
