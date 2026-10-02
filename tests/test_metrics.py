import numpy as np

from chile_forecast.metrics import calc_fit_metrics


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
