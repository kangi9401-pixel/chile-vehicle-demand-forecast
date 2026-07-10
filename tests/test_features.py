import numpy as np
import pandas as pd

from chile_forecast.config import DATE_COL
from chile_forecast.features import build_feature_dataframe, get_longest_non_nan_slice
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


def _resource_price_raw_df():
    dates = pd.date_range("2020-01-31", periods=24, freq="ME")
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        DATE_COL: dates,
        "Middle": 5000.0, "High": 2000.0, "Population": 10000.0, "Interest Rate": 3.0,
        "Crude_Oil_USD_per_Barrel": 50 + rng.normal(size=24),
        "Iron_Ore_USD_per_Ton": 70 + rng.normal(size=24),
        "Copper_USD_per_Ton": 6000 + rng.normal(size=24) * 200,
    })


def test_avg_resource_price_is_invariant_to_a_single_column_rescale():
    """Regression test: an unweighted mean of oil/iron/copper prices is dominated by
    copper's much larger scale, so multiplying copper alone by 100 (e.g. a unit change)
    used to shift avg_resource_price substantially. Z-scoring each series first means
    this rescale shouldn't move the composite at all."""
    raw_df = _resource_price_raw_df()
    raw_df_rescaled = raw_df.copy()
    raw_df_rescaled["Copper_USD_per_Ton"] = raw_df_rescaled["Copper_USD_per_Ton"] * 100

    result = build_feature_dataframe(raw_df)
    result_rescaled = build_feature_dataframe(raw_df_rescaled)

    np.testing.assert_allclose(
        result["avg_resource_price"].values, result_rescaled["avg_resource_price"].values, atol=1e-6,
    )
