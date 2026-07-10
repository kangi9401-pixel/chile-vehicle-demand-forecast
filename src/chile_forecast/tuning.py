"""Hyperparameter search.

XGBoost is tuned with a proper time-series cross-validation (`TimeSeriesSplit`,
expanding-window, no shuffling) via `RandomizedSearchCV`. DeepAR doesn't have an
sklearn-compatible interface, so it's tuned with a small manual grid scored by MAE
on a held-out validation window -- the months immediately preceding the final
holdout, never seen by any model selection or evaluation step downstream.

Both searches run once, on a single representative segment (`Total Market Size`,
the aggregate market), rather than once per segment: with only ~100 monthly
observations per segment, per-segment tuning would overfit the search itself to
each series' noise. The chosen hyperparameters are then reused across all
segments. See docs/REPORT.md for the resulting comparison tables.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from chile_forecast.config import DATE_COL
from chile_forecast.deepar_model import run_deepar_window

logger = logging.getLogger(__name__)

XGB_PARAM_DISTRIBUTIONS: Dict = {
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth": [2, 3, 4, 5],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
}


def tune_xgb_hyperparameters(
    X: pd.DataFrame, y: np.ndarray, n_splits: int = 5, n_iter: int = 25, random_state: int = 42,
) -> Tuple[Dict, pd.DataFrame]:
    """Randomized search over XGBoost hyperparameters using expanding-window CV.

    `TimeSeriesSplit` only ever validates on folds that come *after* their training
    fold in time, so this doesn't leak future data into model selection the way a
    plain k-fold CV would on a time series.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    base_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=random_state)
    search = RandomizedSearchCV(
        base_model,
        param_distributions=XGB_PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        random_state=random_state,
        n_jobs=-1,
    )
    search.fit(X, y)
    results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    results = results[[c for c in results.columns if c.startswith("param_") or c in ("mean_test_score", "std_test_score", "rank_test_score")]]
    return search.best_params_, results


DEEPAR_SEARCH_SPACE: List[Dict] = [
    {"hidden_size": 40, "num_layers": 2, "lr": 1e-3},
    {"hidden_size": 40, "num_layers": 3, "lr": 1e-3},
    {"hidden_size": 64, "num_layers": 2, "lr": 1e-3},
    {"hidden_size": 64, "num_layers": 2, "lr": 5e-4},
]


def tune_deepar_hyperparameters(
    seg: str,
    series_obs: pd.DataFrame,
    final_feature_df_obs: pd.DataFrame,
    val_n: int,
    holdout_n: int,
    search_space: Optional[List[Dict]] = None,
) -> Tuple[Dict, pd.DataFrame]:
    """Grid search over `search_space`, each config scored by MAE on a `val_n`-month
    window that immediately precedes the final `holdout_n`-month test window (so the
    validation window and the eventual test window never overlap)."""
    search_space = search_space or DEEPAR_SEARCH_SPACE
    train_end_date = series_obs[DATE_COL].iloc[-(holdout_n + val_n) - 1]
    val_actual = series_obs[seg].values[-(holdout_n + val_n):len(series_obs) - holdout_n]

    rows = []
    best_params, best_mae = None, np.inf
    for config in search_space:
        result = run_deepar_window(
            seg, series_obs, final_feature_df_obs, train_end_date,
            prediction_length=val_n, max_epochs=15, deepar_kwargs=config,
        )
        mae = mean_absolute_error(val_actual, result["preds_mean"])
        rows.append({**config, "val_MAE": mae})
        logger.info("[%s] DeepAR config %s -> val MAE %.2f", seg, config, mae)
        if mae < best_mae:
            best_mae, best_params = mae, config

    return best_params, pd.DataFrame(rows).sort_values("val_MAE")
