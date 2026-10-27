"""Small, hand-verifiable interaction fixtures."""

import pandas as pd
import pytest


@pytest.fixture
def ratings():
    """Return six unique ratings with deliberately non-contiguous IDs."""
    return pd.DataFrame(
        {
            "userId": [10, 10, 10, 30, 30, 30],
            "movieId": [5, 20, 80, 5, 20, 80],
            "rating": [5.0, 4.0, 3.0, 1.0, 2.0, 3.0],
            "timestamp": [1, 2, 3, 4, 5, 6],
        }
    )
