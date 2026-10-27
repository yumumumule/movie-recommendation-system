"""Recommend unseen catalogue movies using a persisted rating model."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import DEFAULT_DATA, load_data
from src.models import RatingModel, load_model


def recommend(
    model: RatingModel,
    movies: pd.DataFrame,
    user_id: int,
    top_k: int = 10,
    observed: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Rank unseen movies, breaking score ties by ascending original movie ID.

    Exclude the fitted training history and any explicitly supplied observations.
    Callers must not supply held-out interactions during offline ranking evaluation.
    Unknown users receive the model's documented bias/global-mean fallback.
    """
    model._check_fitted()
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not {"movieId", "title", "genres"}.issubset(movies.columns):
        raise ValueError("Catalogue requires movieId, title, genres")
    if movies.movieId.duplicated().any():
        raise ValueError("Catalogue movie IDs must be unique")
    seen = set(model.seen_items[model.seen_users == user_id].tolist())
    if observed is not None:
        seen.update(observed.loc[observed.userId == user_id, "movieId"].tolist())
    candidates = movies.loc[~movies.movieId.isin(seen), ["movieId", "title", "genres"]].copy()
    candidates["predicted_rating"] = model.predict(
        np.full(len(candidates), user_id), candidates.movieId.to_numpy()
    )
    return (
        candidates.sort_values(
            ["predicted_rating", "movieId"], ascending=[False, True], kind="stable"
        )
        .head(top_k)
        .reset_index(drop=True)
    )


def main() -> None:
    """Print recommendations; handle missing artifacts with an actionable error."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--model", type=Path, default=Path("artifacts/model.npz"))
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    try:
        if args.user_id < 1:
            raise ValueError("user-id must be a positive MovieLens ID")
        model = load_model(args.model)
        ratings, movies = load_data(args.data_dir)
        if args.user_id not in model.users.ids:
            print("Unknown user: using item biases/global mean (no personalization).")
        result = recommend(model, movies, args.user_id, args.top_k, observed=ratings)
        print(result.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    except (FileNotFoundError, ValueError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
