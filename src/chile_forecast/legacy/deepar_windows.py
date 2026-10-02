"""Legacy DeepAR wrappers over `train_and_forecast`: the fixed 3-month holdout, the
multi-year forecast, and an arbitrary rolling-origin window (used by the legacy
walk-forward backtest and hyperparameter search).
"""

from typing import Dict, Optional

import pandas as pd

from chile_forecast.config import DATE_COL, FREQ, HOLDOUT_N
from chile_forecast.deepar_model import build_feat_dynamic_real, train_and_forecast


def run_deepar_holdout(
    seg: str,
    series_obs: pd.DataFrame,
    final_feature_df_obs: pd.DataFrame,
    holdout_start: pd.Timestamp,
    deepar_kwargs: Optional[Dict] = None,
) -> Dict:
    """Train on all-but-the-last `HOLDOUT_N` months and predict them, for backtesting."""
    train_target = series_obs[seg].values[:-HOLDOUT_N]
    train_dates = series_obs[DATE_COL].iloc[:-HOLDOUT_N]
    feat_dynamic_real = build_feat_dynamic_real(series_obs[DATE_COL], final_feature_df_obs)

    result = train_and_forecast(
        train_dates, train_target, feat_dynamic_real,
        prediction_length=HOLDOUT_N, max_epochs=20, deepar_kwargs=deepar_kwargs,
    )
    result["future_dates"] = pd.date_range(holdout_start, periods=HOLDOUT_N, freq=FREQ)
    return result


def run_deepar_window(
    seg: str,
    series_obs: pd.DataFrame,
    final_feature_df_obs: pd.DataFrame,
    train_end_date: pd.Timestamp,
    prediction_length: int,
    max_epochs: int = 20,
    deepar_kwargs: Optional[Dict] = None,
) -> Dict:
    """Train on series_obs up to (and including) `train_end_date` and forecast the next
    `prediction_length` months. Used for the rolling-origin backtest and for
    hyperparameter search validation windows -- both stay inside observed history, so
    (unlike the long-term forecast) the evaluation window uses real macro covariates
    rather than a forward-filled assumption about the future."""
    window = series_obs[series_obs[DATE_COL] <= train_end_date]
    train_target = window[seg].values
    train_dates = window[DATE_COL]

    eval_end_idx = len(window) + prediction_length
    full_dates = series_obs[DATE_COL].iloc[:eval_end_idx]
    feat_dynamic_real = build_feat_dynamic_real(full_dates, final_feature_df_obs)

    result = train_and_forecast(
        train_dates, train_target, feat_dynamic_real,
        prediction_length=prediction_length, max_epochs=max_epochs, deepar_kwargs=deepar_kwargs,
    )
    result["future_dates"] = series_obs[DATE_COL].iloc[len(window):eval_end_idx].reset_index(drop=True)
    return result


def run_deepar_longterm(
    seg: str,
    series_obs: pd.DataFrame,
    final_feature_df: pd.DataFrame,
    feature_cols: pd.Index,
    obs_end: pd.Timestamp,
    prediction_end_date: pd.Timestamp,
    deepar_kwargs: Optional[Dict] = None,
) -> Dict:
    """Train on the full observed history and forecast out to `prediction_end_date`.

    Macro covariates beyond `obs_end` don't exist yet, so they're held at their last
    observed value (see `build_feat_dynamic_real`) -- a real assumption, not free
    information, and one worth stating explicitly: this is a "if the economy stayed
    where it last was" scenario, not a macro forecast in its own right.
    """
    train_target = series_obs[seg].values
    train_dates = series_obs[DATE_COL]

    future_start = (obs_end + pd.DateOffset(months=1)) + pd.offsets.MonthEnd(0)
    future_dates = pd.date_range(start=future_start, end=prediction_end_date, freq=FREQ)
    full_dates = pd.concat(
        [train_dates.reset_index(drop=True), pd.Series(future_dates)], ignore_index=True
    )
    feat_dynamic_real = build_feat_dynamic_real(full_dates, final_feature_df)

    result = train_and_forecast(
        train_dates, train_target, feat_dynamic_real,
        prediction_length=len(future_dates), max_epochs=30, deepar_kwargs=deepar_kwargs,
    )
    result["future_dates"] = future_dates
    return result
