"""Final rating, full-catalogue ranking, cold-start and activity evaluations."""

import numpy as np
import pandas as pd

from src.metrics import mae, rmse
from src.models import RatingModel
from src.recommend import recommend


def rating_metrics(model: RatingModel, ratings: pd.DataFrame) -> dict[str, float]:
    """Evaluate clipped rating predictions on a supplied holdout."""
    prediction = model.predict(ratings.userId.to_numpy(), ratings.movieId.to_numpy())
    return {
        "rmse": rmse(ratings.rating.to_numpy(), prediction),
        "mae": mae(ratings.rating.to_numpy(), prediction),
    }


def activity_metrics(model: RatingModel, test: pd.DataFrame) -> pd.DataFrame:
    """Report test errors grouped by training history length, without test-derived bins."""
    counts = dict(zip(model.users.ids, model.user_counts))
    scored = test.copy()
    scored["training_count"] = scored.userId.map(counts).fillna(0)
    scored["prediction"] = model.predict(test.userId.to_numpy(), test.movieId.to_numpy())
    bins = [("0-19", 0, 20), ("20-49", 20, 50), ("50-99", 50, 100), ("100+", 100, np.inf)]
    rows = []
    for name, lower, upper in bins:
        group = scored.loc[(scored.training_count >= lower) & (scored.training_count < upper)]
        if len(group):
            rows.append(
                {
                    "training_history": name,
                    "n_ratings": len(group),
                    "n_users": group.userId.nunique(),
                    "rmse": rmse(group.rating.to_numpy(), group.prediction.to_numpy()),
                    "mae": mae(group.rating.to_numpy(), group.prediction.to_numpy()),
                }
            )
    return pd.DataFrame(rows)


def cold_start_metrics(model: RatingModel, test: pd.DataFrame) -> pd.DataFrame:
    """Separate warm pairs, unknown users, unknown items, and unknown-both pairs."""
    known_u = test.userId.isin(model.users.ids).to_numpy()
    known_i = test.movieId.isin(model.items.ids).to_numpy()
    rows = []
    for name, mask in (
        ("warm", known_u & known_i),
        ("unknown_user", ~known_u & known_i),
        ("unknown_item", known_u & ~known_i),
        ("unknown_both", ~known_u & ~known_i),
    ):
        rows.append(
            {
                "segment": name,
                "n_ratings": int(mask.sum()),
                **(
                    rating_metrics(model, test.loc[mask])
                    if mask.any()
                    else {"rmse": None, "mae": None}
                ),
            }
        )
    return pd.DataFrame(rows)


def ranking_metrics(
    model: RatingModel,
    test: pd.DataFrame,
    movies: pd.DataFrame,
    excluded: pd.DataFrame,
    k: int = 10,
    threshold: float = 4.0,
) -> dict:
    """Macro-average full-catalogue binary relevance metrics for eligible users.

    Relevant items: test ratings >= threshold. Candidates: complete movie catalogue
    excluding train+validation interactions. Unobserved/test-low items are treated
    as nonrelevant, an exposure-biased offline convention. No sampled negatives.
    Users without any relevant test items are excluded and counted explicitly.
    """
    if k < 1:
        raise ValueError("k must be positive")
    relevant = test.loc[test.rating >= threshold].groupby("userId").movieId.apply(set)
    values, candidate_counts = [], []
    for user_id, positives in relevant.items():
        ranked = recommend(model, movies, int(user_id), k, observed=excluded)
        seen = set(model.seen_items[model.seen_users == user_id].tolist())
        seen.update(excluded.loc[excluded.userId == user_id, "movieId"].tolist())
        candidate_counts.append(int((~movies.movieId.isin(seen)).sum()))
        hits = ranked.movieId.isin(positives).to_numpy(dtype=float)
        discount = 1 / np.log2(np.arange(len(hits)) + 2)
        ideal = np.sum(1 / np.log2(np.arange(min(k, len(positives))) + 2))
        values.append(
            [hits.sum() / k, hits.sum() / len(positives), float(np.dot(hits, discount) / ideal)]
        )
    averages = np.mean(values, axis=0).tolist() if values else [None, None, None]
    return {
        "k": k,
        "relevance_threshold": threshold,
        "eligible_users": len(values),
        "excluded_users_without_positive_test": int(test.userId.nunique() - len(values)),
        "mean_candidates": float(np.mean(candidate_counts)) if values else None,
        "precision_at_k": averages[0],
        "recall_at_k": averages[1],
        "ndcg_at_k": averages[2],
    }
