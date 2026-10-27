"""Five explicit-feedback models, including NumPy mini-batch SGD factorization.

No recommender library is used. All prediction methods accept original MovieLens
IDs, and fitted encoders travel with the model in a non-pickle NPZ archive.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import validate_ratings
from src.metrics import rmse
from src.preprocessing import IdEncoder


class RatingModel:
    """Shared fitted state, cold-start behavior, clipping, and serialization."""

    def _initialize(self, ratings: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        validate_ratings(ratings if "timestamp" in ratings else ratings.assign(timestamp=0))
        self.users = IdEncoder(ratings.userId.to_numpy())
        self.items = IdEncoder(ratings.movieId.to_numpy())
        u = self.users.transform(ratings.userId.to_numpy())
        i = self.items.transform(ratings.movieId.to_numpy())
        r = ratings.rating.to_numpy(dtype=float)
        self.global_mean = float(r.mean())
        self.user_bias = np.zeros(len(self.users.ids))
        self.item_bias = np.zeros(len(self.items.ids))
        self.user_factors = np.zeros((len(self.users.ids), 0))
        self.item_factors = np.zeros((len(self.items.ids), 0))
        self.seen_users = ratings.userId.to_numpy(dtype=np.int64).copy()
        self.seen_items = ratings.movieId.to_numpy(dtype=np.int64).copy()
        self.user_counts = np.bincount(u, minlength=len(self.users.ids))
        self.history = []
        self.best_epoch = 0
        return u, i, r

    def _check_fitted(self) -> None:
        if not hasattr(self, "global_mean"):
            raise ValueError("Model is not fitted")

    def _raw_predict(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        self._check_fitted()
        u, i = self.users.transform(user_ids), self.items.transform(item_ids)
        if u.shape != i.shape:
            raise ValueError("User and movie ID arrays must have the same shape")
        known_u, known_i = u >= 0, i >= 0
        prediction = np.full(len(u), self.global_mean)
        prediction[known_u] += self.user_bias[u[known_u]]
        prediction[known_i] += self.item_bias[i[known_i]]
        both = known_u & known_i
        if self.user_factors.shape[1]:
            prediction[both] += np.einsum(
                "ij,ij->i", self.user_factors[u[both]], self.item_factors[i[both]]
            )
        return prediction

    def predict(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        """Predict ratings clipped to MovieLens's fixed [0.5, 5] scale.

        Unknown entities contribute zero bias and zero interaction. A known
        counterpart's fitted bias remains available; two unknowns use the mean.
        """
        return np.clip(self._raw_predict(user_ids, item_ids), 0.5, 5.0)

    def get_params(self) -> dict:
        """Return constructor configuration, independent of fitted arrays."""
        return {}

    def save(self, path: Path) -> None:
        """Persist parameters, raw ID maps and training history without pickle."""
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "format_version": 1,
            "class": type(self).__name__,
            "params": self.get_params(),
            "global_mean": self.global_mean,
            "best_epoch": self.best_epoch,
            "history": self.history,
        }
        # A file handle prevents NumPy silently adding '.npz' to arbitrary paths.
        with path.open("wb") as handle:
            np.savez_compressed(
                handle,
                metadata=json.dumps(metadata),
                users=self.users.ids,
                items=self.items.ids,
                user_bias=self.user_bias,
                item_bias=self.item_bias,
                user_factors=self.user_factors,
                item_factors=self.item_factors,
                seen_users=self.seen_users,
                seen_items=self.seen_items,
                user_counts=self.user_counts,
            )


class GlobalMean(RatingModel):
    """Predict the training mean for every pair."""

    def fit(self, ratings: pd.DataFrame) -> "GlobalMean":
        """Estimate the global training mean."""
        self._initialize(ratings)
        return self


class UserMean(RatingModel):
    """Predict each user's empirical mean, falling back to the global mean."""

    def fit(self, ratings: pd.DataFrame) -> "UserMean":
        """Compute user means using training labels only."""
        u, _, r = self._initialize(ratings)
        self.user_bias = np.bincount(u, weights=r) / self.user_counts - self.global_mean
        return self


class ItemMean(RatingModel):
    """Predict each movie's empirical mean, falling back to the global mean."""

    def fit(self, ratings: pd.DataFrame) -> "ItemMean":
        """Compute item means using training labels only."""
        _, i, r = self._initialize(ratings)
        self.item_bias = np.bincount(i, weights=r) / np.bincount(i) - self.global_mean
        return self


class BiasCF(RatingModel):
    """Regularized additive user/item biases fitted by coordinate descent.

    Minimize sum(error**2) + regularization * (sum(b_u**2) + sum(b_i**2)),
    with the global mean fixed to the training mean.
    """

    def __init__(self, regularization: float = 10.0, epochs: int = 30):
        if not np.isfinite(regularization) or regularization < 0 or epochs < 1:
            raise ValueError("Regularization must be nonnegative and epochs positive")
        self.regularization, self.epochs = regularization, epochs

    def fit(self, ratings: pd.DataFrame) -> "BiasCF":
        """Alternate exact regularized user and item updates."""
        u, i, r = self._initialize(ratings)
        for _ in range(self.epochs):
            self.user_bias = np.bincount(u, weights=r - self.global_mean - self.item_bias[i]) / (
                np.bincount(u) + self.regularization
            )
            self.item_bias = np.bincount(i, weights=r - self.global_mean - self.user_bias[u]) / (
                np.bincount(i) + self.regularization
            )
        return self

    def get_params(self) -> dict:
        """Return coordinate-descent settings."""
        return {"regularization": self.regularization, "epochs": self.epochs}


class MatrixFactorization(RatingModel):
    """Biased matrix factorization trained with shuffled mini-batch SGD.

    Each batch sums gradients computed from a single pre-update snapshot.
    Repeated IDs accumulate with np.add.at rather than losing updates through
    advanced-index assignment. L2 penalties apply per observed interaction.
    The learning rate multiplies summed (not averaged) batch gradients.
    """

    def __init__(
        self,
        n_factors: int = 16,
        learning_rate: float = 0.01,
        regularization: float = 0.1,
        epochs: int = 60,
        batch_size: int = 256,
        patience: int = 7,
        min_delta: float = 1e-5,
        seed: int = 42,
    ):
        if min(n_factors, epochs, batch_size, patience) < 1:
            raise ValueError("Factor count, epochs, batch size and patience must be positive")
        if (
            not np.isfinite([learning_rate, regularization, min_delta]).all()
            or learning_rate <= 0
            or regularization < 0
            or min_delta < 0
        ):
            raise ValueError("Invalid learning rate, regularization or min_delta")
        self.n_factors, self.learning_rate = n_factors, learning_rate
        self.regularization, self.epochs = regularization, epochs
        self.batch_size, self.patience = batch_size, patience
        self.min_delta, self.seed = min_delta, seed

    def _sgd_batch(self, u: np.ndarray, i: np.ndarray, r: np.ndarray) -> None:
        """Apply simultaneous sum-gradient updates for one mini-batch."""
        p, q = self.user_factors[u].copy(), self.item_factors[i].copy()
        bu, bi = self.user_bias[u].copy(), self.item_bias[i].copy()
        error = r - (self.global_mean + bu + bi + np.einsum("ij,ij->i", p, q))
        rate, reg = self.learning_rate, self.regularization
        self.user_bias += rate * np.bincount(
            u, weights=error - reg * bu, minlength=len(self.user_bias)
        )
        self.item_bias += rate * np.bincount(
            i, weights=error - reg * bi, minlength=len(self.item_bias)
        )
        np.add.at(self.user_factors, u, rate * (error[:, None] * q - reg * p))
        np.add.at(self.item_factors, i, rate * (error[:, None] * p - reg * q))

    def fit(
        self,
        ratings: pd.DataFrame,
        validation: pd.DataFrame | None = None,
    ) -> "MatrixFactorization":
        """Fit train-only state; restore the lowest validation-RMSE checkpoint.

        Without validation, run a fixed number of epochs and keep the final
        parameters. Validation is used only to choose stopping time.
        """
        u, i, r = self._initialize(ratings)
        if validation is not None:
            validate_ratings(
                validation if "timestamp" in validation else validation.assign(timestamp=0)
            )
            train_pairs = pd.MultiIndex.from_frame(ratings[["userId", "movieId"]])
            val_pairs = pd.MultiIndex.from_frame(validation[["userId", "movieId"]])
            if len(train_pairs.intersection(val_pairs)):
                raise ValueError("Training and validation user/movie pairs overlap")
        rng = np.random.default_rng(self.seed)
        self.user_factors = rng.normal(0, 0.05, (len(self.users.ids), self.n_factors))
        self.item_factors = rng.normal(0, 0.05, (len(self.items.ids), self.n_factors))
        best_score, patience_score, stale, best_state = np.inf, np.inf, 0, None
        for epoch in range(1, self.epochs + 1):
            order = rng.permutation(len(r))
            for start in range(0, len(order), self.batch_size):
                batch = order[start : start + self.batch_size]
                self._sgd_batch(u[batch], i[batch], r[batch])
            raw = self._raw_predict(ratings.userId.to_numpy(), ratings.movieId.to_numpy())
            penalty = (
                self.user_bias[u] ** 2
                + self.item_bias[i] ** 2
                + np.sum(self.user_factors[u] ** 2, axis=1)
                + np.sum(self.item_factors[i] ** 2, axis=1)
            )
            loss = float(0.5 * np.mean((r - raw) ** 2 + self.regularization * penalty))
            if not np.isfinite(loss):
                raise FloatingPointError("SGD diverged; reduce the learning rate or batch size")
            record = {
                "epoch": epoch,
                "train_loss": loss,
                "train_rmse": rmse(r, np.clip(raw, 0.5, 5.0)),
            }
            if validation is not None:
                score = rmse(
                    validation.rating.to_numpy(),
                    self.predict(validation.userId.to_numpy(), validation.movieId.to_numpy()),
                )
                record["validation_rmse"] = score
                if score < best_score:
                    best_score, self.best_epoch = score, epoch
                    best_state = [
                        a.copy()
                        for a in (
                            self.user_bias,
                            self.item_bias,
                            self.user_factors,
                            self.item_factors,
                        )
                    ]
                if score < patience_score - self.min_delta:
                    patience_score, stale = score, 0
                else:
                    stale += 1
            else:
                self.best_epoch = epoch
            self.history.append(record)
            if validation is not None and stale >= self.patience:
                break
        if best_state is not None:
            self.user_bias, self.item_bias, self.user_factors, self.item_factors = best_state
        return self

    def get_params(self) -> dict:
        """Return all optimization and reproducibility settings."""
        return {
            key: getattr(self, key)
            for key in (
                "n_factors",
                "learning_rate",
                "regularization",
                "epochs",
                "batch_size",
                "patience",
                "min_delta",
                "seed",
            )
        }


def load_model(path: Path) -> RatingModel:
    """Load only known model types from a versioned NPZ archive; disallow pickle."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model missing: {path}. Run python -m src.train first.")
    classes = {
        cls.__name__: cls for cls in (GlobalMean, UserMean, ItemMean, BiasCF, MatrixFactorization)
    }
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"]))
        if metadata.get("format_version") != 1 or metadata.get("class") not in classes:
            raise ValueError("Unsupported model format or model type")
        model = classes[metadata["class"]](**metadata["params"])
        model.users, model.items = IdEncoder(saved["users"]), IdEncoder(saved["items"])
        model.global_mean = metadata["global_mean"]
        model.best_epoch, model.history = metadata["best_epoch"], metadata["history"]
        for name in (
            "user_bias",
            "item_bias",
            "user_factors",
            "item_factors",
            "seen_users",
            "seen_items",
            "user_counts",
        ):
            setattr(model, name, saved[name].copy())
    return model
