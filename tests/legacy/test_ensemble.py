import numpy as np
import pandas as pd
import pytest

from chile_forecast.legacy.ensemble import build_holdout_row, compute_ensemble_weights


def test_compute_ensemble_weights_equal_mae_gives_half_half():
    rolling_df = pd.DataFrame({
        "Segment": ["A", "A"],
        "MAE_DeepAR": [10.0, 10.0],
        "MAE_XGBoost": [10.0, 10.0],
    })
    weights = compute_ensemble_weights(rolling_df)
    assert weights["A"] == pytest.approx(0.5)


def test_compute_ensemble_weights_favors_more_accurate_model():
    rolling_df = pd.DataFrame({
        "Segment": ["B", "B"],
        "MAE_DeepAR": [5.0, 5.0],
        "MAE_XGBoost": [20.0, 20.0],
    })
    weights = compute_ensemble_weights(rolling_df)
    assert weights["B"] > 0.5  # DeepAR was more accurate, so it gets more weight


def test_compute_ensemble_weights_empty_df_returns_empty_dict():
    assert compute_ensemble_weights(pd.DataFrame()) == {}


def test_build_holdout_row_weight_zero_uses_xgb_only():
    y_test = np.array([100.0, 110.0, 120.0])
    preds_deepar = np.array([0.0, 0.0, 0.0])
    preds_xgb = np.array([100.0, 110.0, 120.0])
    row = build_holdout_row(
        "seg", y_test, preds_deepar, preds_xgb, pd.Timestamp("2025-01-31"), weight_deepar=0.0,
    )
    assert row["MAE_Ensemble"] == pytest.approx(row["MAE_XGBoost"])
    assert row["Weight_DeepAR"] == 0.0


def test_build_holdout_row_default_weight_is_fifty_fifty():
    y_test = np.array([100.0, 100.0, 100.0])
    preds_deepar = np.array([90.0, 90.0, 90.0])
    preds_xgb = np.array([110.0, 110.0, 110.0])
    row = build_holdout_row("seg", y_test, preds_deepar, preds_xgb, pd.Timestamp("2025-01-31"))
    assert row["MAE_Ensemble"] == pytest.approx(0.0)
