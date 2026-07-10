"""DeepAR training utilities.

`train_and_forecast` is the single low-level trainer; `run_deepar_holdout`,
`run_deepar_longterm`, and `run_deepar_window` are thin wrappers over it for the
three call sites this project needs: the fixed 3-month holdout, the multi-year
forecast, and an arbitrary rolling-origin window (used by both the walk-forward
backtest and the hyperparameter search).
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from gluonts.dataset.common import ListDataset
from gluonts.torch.model.deepar import DeepAREstimator

from chile_forecast.config import DATE_COL, FORECAST_STEP, FREQ, HOLDOUT_N, MACRO_COLS_FOR_DEEPAR
from chile_forecast.features import create_time_features

# torch >=2.6 defaults `torch.load(weights_only=True)`, which rejects the
# assorted non-tensor objects (functools.partial, getattr, ...) gluonts embeds
# in its Lightning checkpoints. Lightning's own loader passes `weights_only=None`
# explicitly, which torch treats as "use the restrictive default" -- so a plain
# functools.partial override gets shadowed by that explicit None. We instead
# wrap torch.load to force False whenever the caller didn't ask for True. These
# checkpoints are produced locally by this same run and immediately reloaded,
# so trusting them is safe.
_original_torch_load = torch.load


def _unrestricted_torch_load(*args, **kwargs):
    if kwargs.get("weights_only") is not True:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _unrestricted_torch_load

logger = logging.getLogger(__name__)


def build_feat_dynamic_real(
    full_dates: pd.Series, feature_source_df: pd.DataFrame, macro_cols: List[str] = MACRO_COLS_FOR_DEEPAR,
) -> np.ndarray:
    """Align macro covariates to `full_dates` and stack them with calendar time features.

    `feature_source_df` only needs to cover the dates it actually has data for -- any
    date in `full_dates` beyond that (e.g. a forecast horizon past the last observed
    macro reading) is forward-filled from the last known value. For windows fully
    inside `feature_source_df`'s range (holdout/backtest evaluation), this is an exact
    merge with no filling, i.e. the model is scored against real macro data, not an
    assumption about the future.
    """
    feature_cols = feature_source_df.columns.drop(DATE_COL)
    aligned = pd.DataFrame({DATE_COL: full_dates}).merge(feature_source_df, on=DATE_COL, how="left")
    aligned[feature_cols] = aligned[feature_cols].ffill().fillna(0)
    aligned.replace([np.inf, -np.inf], 0, inplace=True)

    macro_feats = aligned[macro_cols].T.values
    time_feats = create_time_features(full_dates)
    return np.concatenate([macro_feats, time_feats], axis=0)


def train_and_forecast(
    train_dates: pd.Series,
    train_target: np.ndarray,
    feat_dynamic_real_full: np.ndarray,
    prediction_length: int,
    max_epochs: int = 20,
    deepar_kwargs: Optional[Dict] = None,
) -> Dict:
    """Fit DeepAR on (train_dates, train_target) and forecast `prediction_length` steps
    beyond it. `feat_dynamic_real_full` must have `len(train_target) + prediction_length`
    columns (covariates for the training span plus the forecast horizon)."""
    deepar_kwargs = dict(deepar_kwargs or {})

    train_ds = ListDataset(
        [{
            "start": train_dates.iloc[0],
            "target": train_target,
            "feat_dynamic_real": feat_dynamic_real_full[:, :len(train_target)],
        }],
        freq=FREQ,
    )
    test_ds = ListDataset(
        [{
            "start": train_dates.iloc[0],
            "target": train_target,
            "feat_dynamic_real": feat_dynamic_real_full,
        }],
        freq=FREQ,
    )

    estimator = DeepAREstimator(
        freq=FREQ,
        prediction_length=prediction_length,
        context_length=min(len(train_target), FORECAST_STEP * 2),
        trainer_kwargs={
            "max_epochs": max_epochs, "logger": False, "enable_progress_bar": False,
            "accelerator": "cpu",
        },
        **deepar_kwargs,
    )
    predictor = estimator.train(train_ds)
    forecast = next(iter(predictor.predict(test_ds)))

    return {
        "preds_mean": forecast.mean,
        "preds_low": forecast.quantile(0.10),
        "preds_up": forecast.quantile(0.90),
    }


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
