"""Forecast fit metrics."""

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calc_fit_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    if len(y_true) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "R2": np.nan}

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5

    mape_mask = y_true != 0
    mape = (
        np.mean(np.abs((y_true[mape_mask] - y_pred[mape_mask]) / y_true[mape_mask])) * 100
        if np.any(mape_mask)
        else np.nan
    )

    r2 = r2_score(y_true, y_pred) if len(y_true) >= 2 else np.nan

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}
