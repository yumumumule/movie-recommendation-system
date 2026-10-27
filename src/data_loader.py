"""Download official MovieLens data and reject malformed input explicitly."""

import hashlib
import json
import shutil
import ssl
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DEFAULT_DATA = Path("data/raw/ml-latest-small")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 checksum of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_data(destination: Path = DEFAULT_DATA) -> Path:
    """Fetch the official archive atomically; extract only three expected files."""
    destination = Path(destination)
    if all((destination / name).is_file() for name in ("ratings.csv", "movies.csv")):
        load_data(destination)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as folder:
        archive = Path(folder) / "movielens.zip"
        request = urllib.request.Request(
            DATA_URL, headers={"User-Agent": "MovieLensPortfolio/1.0"}
        )
        context = ssl.create_default_context()
        # This CA chain is scoped to the official download request. Some OS
        # stores have not yet incorporated GroupLens's 2026 emSign root.
        context.load_verify_locations(cafile=str(Path(__file__).parent / "certs/grouplens.pem"))
        try:
            with (
                urllib.request.urlopen(request, timeout=120, context=context) as response,
                archive.open("wb") as out,
            ):
                shutil.copyfileobj(response, out)
        except urllib.error.URLError as error:
            raise ConnectionError(
                "Could not securely download GroupLens data. Check network access and "
                "your certificate store, or set SSL_CERT_FILE to your trusted CA bundle."
            ) from error
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as bundle:
            if bundle.testzip() is not None:
                raise ValueError("MovieLens archive failed its CRC integrity check")
            for name in ("ratings.csv", "movies.csv", "README.txt"):
                payload = bundle.read(f"ml-latest-small/{name}")
                (destination / name).write_bytes(payload)
        manifest = {"url": DATA_URL, "archive_sha256": sha256_file(archive)}
        (destination / "download.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
    load_data(destination)
    return destination


def validate_ratings(ratings: pd.DataFrame) -> None:
    """Reject null, duplicate, non-finite, or out-of-domain rating records."""
    required = ["userId", "movieId", "rating", "timestamp"]
    if not set(required).issubset(ratings.columns) or ratings.empty:
        raise ValueError("Ratings must be nonempty and contain " + ", ".join(required))
    values = ratings[required].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Ratings contain missing or non-finite values")
    for column in ("userId", "movieId", "timestamp"):
        data = ratings[column].to_numpy(dtype=float)
        if (data < (0 if column == "timestamp" else 1)).any() or (data != np.floor(data)).any():
            raise ValueError(f"{column} must contain valid integer identifiers/timestamps")
    scores = ratings.rating.to_numpy()
    if ((scores < 0.5) | (scores > 5) | (scores * 2 != np.floor(scores * 2))).any():
        raise ValueError("MovieLens ratings must be half-star values from 0.5 through 5")
    if ratings.duplicated(["userId", "movieId"]).any():
        raise ValueError("Duplicate user/movie pairs would invalidate splitting")


def load_data(directory: Path = DEFAULT_DATA) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load ratings and catalogue, preserving raw IDs; fail rather than silently impute."""
    directory = Path(directory)
    if not all((directory / name).exists() for name in ("ratings.csv", "movies.csv")):
        raise FileNotFoundError("Dataset missing. Run: python -m scripts.download_data")
    ratings = pd.read_csv(directory / "ratings.csv")
    movies = pd.read_csv(directory / "movies.csv")
    validate_ratings(ratings)
    if not {"movieId", "title", "genres"}.issubset(movies.columns) or movies.empty:
        raise ValueError("Movie catalogue requires movieId, title and genres")
    if movies[["movieId", "title", "genres"]].isna().any().any():
        raise ValueError("Movie catalogue contains missing fields")
    ids = pd.to_numeric(movies.movieId, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(ids).all() or (ids <= 0).any() or (ids != np.floor(ids)).any():
        raise ValueError("Movie IDs must be positive integers")
    if movies.movieId.duplicated().any() or not ratings.movieId.isin(movies.movieId).all():
        raise ValueError("Movie catalogue contains duplicates or missing rated movies")
    return ratings, movies
