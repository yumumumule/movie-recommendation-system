"""Input integrity, download extraction, split isolation and train-only encoding."""

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from src.data_loader import download_data, load_data, validate_ratings
from src.preprocessing import IdEncoder, split_ratings


def test_split_is_disjoint_exhaustive_and_reproducible(ratings):
    split = split_ratings(ratings, validation_fraction=0.2, test_fraction=0.2)
    again = split_ratings(ratings, validation_fraction=0.2, test_fraction=0.2)
    pairs = [set(zip(frame.userId, frame.movieId)) for frame in split]
    assert all(not pairs[a] & pairs[b] for a, b in ((0, 1), (0, 2), (1, 2)))
    assert set.union(*pairs) == set(zip(ratings.userId, ratings.movieId))
    for a, b in zip(split, again):
        pd.testing.assert_frame_equal(a, b)
    split[1].loc[:, "rating"] = 0.5
    pd.testing.assert_frame_equal(split[0], again[0])


def test_encoder_uses_training_ids_only():
    encoder = IdEncoder(np.array([100, 5, 100]))
    np.testing.assert_array_equal(encoder.transform([5, 100, 9, 1000]), [0, 1, -1, -1])
    np.testing.assert_array_equal(encoder.ids, [5, 100])


@pytest.mark.parametrize("bad", ["null", "infinite", "range", "duplicate", "fractional_id"])
def test_invalid_data_rejected(ratings, bad):
    frame = ratings.copy()
    if bad == "null":
        frame.loc[0, "rating"] = np.nan
    elif bad == "infinite":
        frame.loc[0, "rating"] = np.inf
    elif bad == "range":
        frame.loc[0, "rating"] = 6
    elif bad == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    else:
        frame["userId"] = frame.userId.astype(float)
        frame.loc[0, "userId"] = 1.5
    with pytest.raises(ValueError):
        validate_ratings(frame)


def test_downloader_extracts_only_expected_members(ratings, tmp_path, monkeypatch):
    content = io.BytesIO()
    movies = pd.DataFrame(
        {"movieId": [5, 20, 80], "title": ["A", "B", "C"], "genres": ["Drama", "Comedy", "Action"]}
    )
    with zipfile.ZipFile(content, "w") as bundle:
        bundle.writestr("ml-latest-small/ratings.csv", ratings.to_csv(index=False))
        bundle.writestr("ml-latest-small/movies.csv", movies.to_csv(index=False))
        bundle.writestr("ml-latest-small/README.txt", "Synthetic fixture")
        bundle.writestr("../../escaped.txt", "must not be extracted")
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *args, **kwargs: io.BytesIO(content.getvalue())
    )
    target = download_data(tmp_path / "data")
    loaded, _ = load_data(target)
    pd.testing.assert_frame_equal(loaded, ratings)
    assert (target / "download.json").exists()
    assert not (tmp_path / "escaped.txt").exists()
    assert download_data(target) == target


def test_missing_dataset(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data"):
        load_data(tmp_path)
