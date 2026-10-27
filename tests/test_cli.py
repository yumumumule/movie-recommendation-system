"""Command-line integration checks using synthetic files, without network access."""

import os
import subprocess
import sys

import pandas as pd

from src.models import ItemMean


def test_recommend_cli_handles_unicode_and_missing_model(tmp_path):
    ratings = pd.DataFrame(
        {"userId": [1, 2], "movieId": [10, 20], "rating": [3.0, 5.0], "timestamp": [1, 2]}
    )
    movies = pd.DataFrame(
        {"movieId": [10, 20], "title": ["Seen", "Höstsonaten"], "genres": ["Drama", "Drama"]}
    )
    ratings.to_csv(tmp_path / "ratings.csv", index=False)
    movies.to_csv(tmp_path / "movies.csv", index=False)
    model_path = tmp_path / "model.npz"
    ItemMean().fit(ratings).save(model_path)
    command = [
        sys.executable,
        "-m",
        "src.recommend",
        "--user-id",
        "1",
        "--data-dir",
        str(tmp_path),
        "--model",
        str(model_path),
    ]
    result = subprocess.run(
        command, capture_output=True, env={**os.environ, "PYTHONIOENCODING": "ascii"}
    )
    assert result.returncode == 0
    assert "Höstsonaten" in result.stdout.decode("utf-8")
    model_path.unlink()
    result = subprocess.run(command, capture_output=True)
    assert result.returncode == 2
    assert b"python -m src.train" in result.stderr
