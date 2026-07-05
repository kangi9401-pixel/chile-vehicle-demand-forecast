import numpy as np
import pandas as pd

from chile_forecast.features import get_longest_non_nan_slice
from chile_forecast.metrics import calc_fit_metrics


def test_longest_non_nan_slice_picks_the_bigger_run():
    series = pd.Series([np.nan, 1, 2, np.nan, 3, 4, 5, 6, np.nan])
    start, end, length = get_longest_non_nan_slice(series)
    assert (start, end, length) == (4, 8, 4)


def test_longest_non_nan_slice_all_valid():
    series = pd.Series([1, 2, 3, 4])
    start, end, length = get_longest_non_nan_slice(series)
    assert (start, end, length) == (0, 4, 4)


def test_longest_non_nan_slice_all_nan():
    series = pd.Series([np.nan, np.nan])
    start, end, length = get_longest_non_nan_slice(series)
    assert length == 0


def test_calc_fit_metrics_perfect_prediction():
    y = [10.0, 20.0, 30.0]
    metrics = calc_fit_metrics(y, y)
    assert metrics["MAE"] == 0
    assert metrics["RMSE"] == 0
    assert metrics["MAPE"] == 0
    assert metrics["R2"] == 1


def test_calc_fit_metrics_ignores_nan_pairs():
    y_true = [10.0, np.nan, 30.0]
    y_pred = [10.0, 99.0, 30.0]
    metrics = calc_fit_metrics(y_true, y_pred)
    assert metrics["MAE"] == 0


def test_calc_fit_metrics_empty_returns_nan():
    metrics = calc_fit_metrics([], [])
    assert np.isnan(metrics["MAE"])
