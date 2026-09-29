"""Forecast fit metrics."""

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calc_fit_metrics(y_true, y_pred, mase_scale: float = np.nan) -> Dict[str, float]:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    if len(y_true) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "WAPE": np.nan, "sMAPE": np.nan, "MASE": np.nan, "R2": np.nan}

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5

    abs_error = np.abs(y_true - y_pred)
    actual_sum = np.abs(y_true).sum()
    wape = float(abs_error.sum() / actual_sum * 100) if actual_sum > 1e-12 else np.nan
    smape_denom = np.abs(y_true) + np.abs(y_pred)
    smape = float(np.mean(np.divide(2 * abs_error, smape_denom, out=np.zeros_like(abs_error), where=smape_denom > 1e-12)) * 100)
    mape_mask = np.abs(y_true) > 1e-12
    mape = float(np.mean(abs_error[mape_mask] / np.abs(y_true[mape_mask])) * 100) if np.any(mape_mask) else np.nan
    mase = float(mae / mase_scale) if np.isfinite(mase_scale) and mase_scale > 1e-12 else np.nan

    r2 = r2_score(y_true, y_pred) if len(y_true) >= 2 else np.nan

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "WAPE": wape, "sMAPE": smape, "MASE": mase, "R2": r2}


def seasonal_mase_scale(train, season_length: int = 12) -> float:
    """Mean absolute seasonal-naive in-sample error used as MASE denominator."""
    values = np.asarray(train, dtype=float)
    if len(values) <= season_length:
        return np.nan
    return float(np.mean(np.abs(values[season_length:] - values[:-season_length])))
