"""Rebuild the selected model without consulting test labels."""

import argparse
import json
from pathlib import Path

from src.data_loader import DEFAULT_DATA, load_data
from src.models import BiasCF, GlobalMean, ItemMean, MatrixFactorization, UserMean
from src.preprocessing import split_ratings


def main() -> None:
    """Train using frozen selection settings; optionally refit for serving on all data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--selection", type=Path, default=Path("results/selection.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/model.npz"))
    parser.add_argument(
        "--all-data",
        action="store_true",
        help="Serving refit only; never use this model for reported test metrics",
    )
    args = parser.parse_args()
    try:
        selection = json.loads(args.selection.read_text(encoding="utf-8"))
        ratings, _ = load_data(args.data_dir)
        train = ratings if args.all_data else split_ratings(ratings, selection["seed"])[0]
        name = selection["selected_model"]
        if name == "Matrix Factorization":
            params = dict(selection["matrix_factorization"]["params"])
            params["epochs"] = selection["matrix_factorization"]["best_epoch"]
            model = MatrixFactorization(**params)
        elif name == "Bias CF":
            model = BiasCF(**selection["bias_cf"])
        else:
            model = {"Global Mean": GlobalMean, "User Mean": UserMean, "Item Mean": ItemMean}[
                name
            ]()
        model.fit(train)
        model.save(args.output)
        print(f"Saved {name}, fitted on {len(train):,} ratings, to {args.output}")
    except (FileNotFoundError, ValueError, KeyError) as error:
        parser.exit(2, f"Error: {error}. Run python -m experiments.run_experiments if needed.\n")


if __name__ == "__main__":
    main()
