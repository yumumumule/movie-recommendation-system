"""Run train/validation selection, freeze choices, then evaluate test once."""

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data_loader import DEFAULT_DATA, download_data, load_data, sha256_file
from src.evaluate import activity_metrics, cold_start_metrics, ranking_metrics, rating_metrics
from src.models import BiasCF, GlobalMean, ItemMean, MatrixFactorization, UserMean
from src.preprocessing import split_ratings
from src.recommend import recommend
from src.visualization import create_figures


def write_json(path: Path, value: dict) -> None:
    """Write readable, finite JSON."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(data_dir: Path, output: Path, artifacts: Path, seed: int = 42) -> None:
    """Execute a predeclared CPU-only experiment grid with a single final test phase."""
    output, artifacts = Path(output), Path(artifacts)
    if (output / "metrics.csv").exists():
        raise FileExistsError("Results already exist; use a new --output-dir for a fresh run")
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    download_data(data_dir)
    ratings, movies = load_data(data_dir)
    train, validation, test = split_ratings(ratings, seed)
    plan = {
        "seed": seed,
        "split": "global random interaction holdout, 70/15/15 (floored holdouts)",
        "n_factors": [8, 16, 32],
        "regularization": [0.02, 0.1, 0.3],
        "learning_rate": [0.005, 0.01],
        "max_epochs": 60,
        "batch_size": 256,
        "patience": 7,
        "min_delta": 1e-5,
        "bias_regularization": [1.0, 5.0, 10.0, 20.0],
        "rating_clip": [0.5, 5.0],
        "ranking_k": 10,
        "relevance_threshold": 4.0,
        "selection_metric": "validation RMSE; no train+validation refit for reported test scores",
    }
    write_json(output / "experiment_plan.json", plan)
    models = {
        "Global Mean": GlobalMean().fit(train),
        "User Mean": UserMean().fit(train),
        "Item Mean": ItemMean().fit(train),
    }
    bias_trials, best_bias, best_bias_score = [], None, float("inf")
    for reg in plan["bias_regularization"]:
        candidate = BiasCF(regularization=reg).fit(train)
        score = rating_metrics(candidate, validation)["rmse"]
        bias_trials.append({"regularization": reg, "validation_rmse": score})
        if score < best_bias_score:
            best_bias, best_bias_score = candidate, score
    models["Bias CF"] = best_bias
    pd.DataFrame(bias_trials).to_csv(output / "bias_trials.csv", index=False)

    trials, best_mf, best_mf_score = [], None, float("inf")
    grid = list(
        itertools.product(plan["n_factors"], plan["regularization"], plan["learning_rate"])
    )
    for trial_id, (factors, reg, rate) in enumerate(grid, 1):
        trial_start = time.perf_counter()
        candidate = MatrixFactorization(
            n_factors=factors,
            regularization=reg,
            learning_rate=rate,
            epochs=plan["max_epochs"],
            batch_size=plan["batch_size"],
            patience=plan["patience"],
            min_delta=plan["min_delta"],
            seed=seed,
        ).fit(train, validation)
        score = rating_metrics(candidate, validation)["rmse"]
        trial = {
            "trial": trial_id,
            "n_factors": factors,
            "regularization": reg,
            "learning_rate": rate,
            "validation_rmse": score,
            "best_epoch": candidate.best_epoch,
            "epochs_run": len(candidate.history),
            "seconds": time.perf_counter() - trial_start,
        }
        trials.append(trial)
        print(
            f"MF {trial_id:2}/{len(grid)}: factors={factors}, reg={reg}, lr={rate}, "
            f"val={score:.5f}, best_epoch={candidate.best_epoch}",
            flush=True,
        )
        pd.DataFrame(trials).to_csv(output / "mf_trials.csv", index=False)
        if score < best_mf_score:
            best_mf, best_mf_score = candidate, score
    models["Matrix Factorization"] = best_mf
    validation_scores = {
        name: rating_metrics(model, validation)["rmse"] for name, model in models.items()
    }
    champion_name = min(validation_scores, key=validation_scores.get)
    champion = models[champion_name]
    selection = {
        "selected_model": champion_name,
        "validation_rmse": validation_scores,
        "seed": seed,
        "matrix_factorization": {"params": best_mf.get_params(), "best_epoch": best_mf.best_epoch},
        "bias_cf": best_bias.get_params(),
        "protocol": "All choices frozen before test rating or ranking evaluation",
    }
    # Freeze settings and save models before accessing any test labels for scoring.
    write_json(output / "selection.json", selection)
    champion.save(artifacts / "model.npz")
    best_mf.save(artifacts / "matrix_factorization.npz")
    pd.DataFrame(best_mf.history).to_csv(output / "training_history.csv", index=False)
    print(f"Selection frozen: {champion_name}. Beginning final test evaluation.", flush=True)

    metrics, rankings = [], []
    excluded = pd.concat([train, validation], ignore_index=True)
    for name, model in models.items():
        test_score = rating_metrics(model, test)
        metrics.append(
            {
                "model": name,
                "validation_rmse": validation_scores[name],
                "test_rmse": test_score["rmse"],
                "test_mae": test_score["mae"],
            }
        )
        rankings.append({"model": name, **ranking_metrics(model, test, movies, excluded)})
    pd.DataFrame(metrics).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(rankings).to_csv(output / "ranking_metrics.csv", index=False)
    activity_metrics(champion, test).to_csv(output / "user_activity.csv", index=False)
    cold_start_metrics(champion, test).to_csv(output / "cold_start.csv", index=False)
    recommend(champion, movies, 1, 10, observed=ratings).to_csv(
        output / "recommendations_user_1.csv", index=False
    )
    provenance = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name) for name in ("numpy", "pandas", "matplotlib")
        },
        "seed": seed,
        "n_ratings": len(ratings),
        "n_users": ratings.userId.nunique(),
        "catalogue_movies": len(movies),
        "rated_movies": ratings.movieId.nunique(),
        "split_sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
        "input_sha256": {
            name: sha256_file(data_dir / name) for name in ("ratings.csv", "movies.csv")
        },
        "split_sha256": {
            name: hashlib.sha256(
                frame[["userId", "movieId", "rating", "timestamp"]]
                .to_csv(index=False, lineterminator="\n")
                .encode()
            ).hexdigest()
            for name, frame in (("train", train), ("validation", validation), ("test", test))
        },
        "git_base_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip(),
        "training_and_evaluation_seconds": time.perf_counter() - started,
        "split_hash_format": "UTF-8 CSV, columns userId/movieId/rating/timestamp, LF line endings",
    }
    download_manifest = Path(data_dir) / "download.json"
    if download_manifest.exists():
        provenance["archive_sha256"] = json.loads(download_manifest.read_text(encoding="utf-8"))[
            "archive_sha256"
        ]
    provenance["n_users"] = int(provenance["n_users"])
    provenance["rated_movies"] = int(provenance["rated_movies"])
    write_json(output / "provenance.json", provenance)
    create_figures(output, selection)
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)


def main() -> None:
    """Run experiments with portable relative paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.data_dir, args.output_dir, args.artifacts_dir, args.seed)


if __name__ == "__main__":
    main()
