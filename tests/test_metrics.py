"""Independent arithmetic and invalid-input checks."""

import numpy as np
import pytest

from src.metrics import mae, rmse


def test_hand_calculated_metrics():
    assert rmse([1, 3, 5], [2, 3, 3]) == pytest.approx(np.sqrt(5 / 3))
    assert mae([1, 3, 5], [2, 3, 3]) == pytest.approx(1)


@pytest.mark.parametrize("metric", [rmse, mae])
@pytest.mark.parametrize(
    "actual,predicted", [([], []), ([1], [1, 2]), ([np.nan], [1]), ([1], [np.inf]), ([[1]], [[1]])]
)
def test_invalid_metric_inputs(metric, actual, predicted):
    with pytest.raises(ValueError):
        metric(actual, predicted)
