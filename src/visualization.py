"""Render experiment artifacts into compact, publication-style PNG figures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def create_figures(output: Path, selected: dict) -> None:
    """Plot six analyses using saved measured CSVs, never synthetic results."""
    output = Path(output)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 120}
    )

    def save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(figures / f"{name}.png", dpi=170, bbox_inches="tight")
        plt.close()

    metrics = pd.read_csv(output / "metrics.csv")
    plt.figure(figsize=(8, 4))
    bars = plt.bar(metrics.model, metrics.test_rmse, color="#28679b", width=0.65)
    plt.bar_label(bars, fmt="%.3f", padding=3)
    plt.ylim(0, metrics.test_rmse.max() * 1.15)
    plt.ylabel("Test RMSE (stars)")
    plt.title("Rating prediction on the untouched test split")
    plt.xticks(rotation=15, ha="right")
    save("model_comparison")

    history = pd.read_csv(output / "training_history.csv")
    for column, label, title, filename in (
        (
            "train_loss",
            "Regularized training objective",
            "Matrix factorization training",
            "training_curve",
        ),
        (
            "validation_rmse",
            "Validation RMSE (stars)",
            "Validation and early stopping",
            "validation_curve",
        ),
    ):
        plt.figure(figsize=(7, 4))
        plt.plot(history.epoch, history[column], color="#28679b", linewidth=2)
        plt.axvline(
            selected["matrix_factorization"]["best_epoch"],
            color="#d27729",
            linestyle="--",
            label="Restored checkpoint",
        )
        plt.xlabel("Epoch")
        plt.ylabel(label)
        plt.title(title)
        plt.legend(frameon=False)
        save(filename)

    trials = pd.read_csv(output / "mf_trials.csv")
    params = selected["matrix_factorization"]["params"]
    for parameter, xlabel, filename in (
        ("n_factors", "Latent dimensions", "latent_dimensions"),
        ("regularization", "Per-interaction L2 strength", "regularization"),
    ):
        curve = trials.loc[trials.learning_rate == params["learning_rate"]]
        other = "regularization" if parameter == "n_factors" else "n_factors"
        curve = curve.loc[curve[other] == params[other]].sort_values(parameter)
        plt.figure(figsize=(7, 4))
        plt.plot(curve[parameter], curve.validation_rmse, "o-", color="#28679b")
        plt.xlabel(xlabel)
        plt.ylabel("Best-epoch validation RMSE (stars)")
        plt.title(f"Sensitivity: {other}={params[other]}, lr={params['learning_rate']}")
        if parameter == "regularization":
            plt.xscale("log")
            plt.xticks(curve[parameter], [f"{value:g}" for value in curve[parameter]])
            plt.minorticks_off()
        else:
            plt.xticks(curve[parameter])
        save(filename)

    activity = pd.read_csv(output / "user_activity.csv")
    plt.figure(figsize=(7, 4))
    bars = plt.bar(activity.training_history, activity.rmse, color="#28679b", width=0.6)
    plt.bar_label(bars, labels=[f"n={n:,}" for n in activity.n_ratings], padding=4, fontsize=9)
    plt.ylim(0, activity.rmse.max() * 1.2)
    plt.xlabel("User's training interactions")
    plt.ylabel("Test RMSE (stars)")
    plt.title("Selected model: error by user activity")
    save("user_activity")
