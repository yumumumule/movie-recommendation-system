"""Ranking order, ties, exclusions, cold start and full-catalogue evaluation."""

import numpy as np
import pandas as pd
import pytest

from src.evaluate import ranking_metrics
from src.models import ItemMean
from src.recommend import recommend


def catalogue():
    return pd.DataFrame(
        {"movieId": [10, 20, 30, 40], "title": ["A", "B", "C", "D"], "genres": ["Drama"] * 4}
    )


def training():
    return pd.DataFrame({"userId": [1, 2, 2], "movieId": [10, 20, 30], "rating": [1.0, 4.0, 5.0]})


def test_order_and_history_exclusion():
    model = ItemMean().fit(training())
    result = recommend(model, catalogue(), 1, 10)
    assert result.movieId.tolist() == [30, 20, 40]
    assert result.predicted_rating.is_monotonic_decreasing
    extra = pd.DataFrame({"userId": [1], "movieId": [30]})
    assert recommend(model, catalogue(), 1, 10, extra).movieId.tolist() == [20, 40]
    assert np.isfinite(recommend(model, catalogue(), 999).predicted_rating).all()


def test_ties_empty_and_invalid_k():
    model = ItemMean().fit(training())
    movies = pd.DataFrame({"movieId": [80, 70], "title": ["Z", "Y"], "genres": ["Drama"] * 2})
    assert recommend(model, movies, 1).movieId.tolist() == [70, 80]
    assert recommend(model, catalogue().iloc[:0], 1).empty
    with pytest.raises(ValueError):
        recommend(model, movies, 1, 0)


def test_full_catalogue_ranking_hand_calculation():
    train = training()
    model = ItemMean().fit(train)
    test = pd.DataFrame({"userId": [1, 1, 2], "movieId": [20, 40, 40], "rating": [4.0, 5.0, 2.0]})
    metrics = ranking_metrics(model, test, catalogue(), train, k=2)
    assert metrics["eligible_users"] == 1
    assert metrics["excluded_users_without_positive_test"] == 1
    assert metrics["mean_candidates"] == 3
    assert metrics["precision_at_k"] == 0.5
    assert metrics["recall_at_k"] == 0.5
    discount = 1 / np.log2(3)
    assert metrics["ndcg_at_k"] == pytest.approx(discount / (1 + discount))
