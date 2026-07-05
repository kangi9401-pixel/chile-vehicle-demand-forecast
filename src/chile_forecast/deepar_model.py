"""DeepAR training/prediction for both the short holdout backtest and the long-term forecast."""

import logging
from typing import Dict

import numpy as np
import pandas as pd
import torch
from gluonts.dataset.common import ListDataset
from gluonts.torch.model.deepar import DeepAREstimator

from chile_forecast.config import (
    DATE_COL, FORECAST_STEP, FREQ, HOLDOUT_N, MACRO_COLS_FOR_DEEPAR,
)
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


def run_deepar_holdout(
    seg: str,
    series_obs: pd.DataFrame,
    final_feature_df_obs: pd.DataFrame,
    holdout_start: pd.Timestamp,
) -> Dict:
    """Train DeepAR on all-but-the-last `HOLDOUT_N` months and predict them, for backtesting."""
    train_target = series_obs[seg].values[:-HOLDOUT_N]
    full_dates = series_obs[DATE_COL]

    macro_feats = final_feature_df_obs.loc[
        final_feature_df_obs[DATE_COL].isin(full_dates), MACRO_COLS_FOR_DEEPAR
    ].T.values
    time_feats = create_time_features(full_dates)
    feat_dynamic_real = np.concatenate([macro_feats, time_feats], axis=0)

    train_ds = ListDataset(
        [{
            "start": series_obs[DATE_COL].iloc[0],
            "target": train_target,
            "feat_dynamic_real": feat_dynamic_real[:, :-HOLDOUT_N],
        }],
        freq=FREQ,
    )
    test_ds = ListDataset(
        [{
            "start": series_obs[DATE_COL].iloc[0],
            "target": train_target,
            "feat_dynamic_real": feat_dynamic_real,
        }],
        freq=FREQ,
    )

    estimator = DeepAREstimator(
        freq=FREQ,
        prediction_length=HOLDOUT_N,
        context_length=min(len(train_target), FORECAST_STEP * 2),
        trainer_kwargs={
            "max_epochs": 20, "logger": False, "enable_progress_bar": False,
            "accelerator": "cpu",
        },
    )
    predictor = estimator.train(train_ds)
    forecast = next(iter(predictor.predict(test_ds)))

    return {
        "preds_mean": forecast.mean,
        "future_dates": pd.date_range(holdout_start, periods=HOLDOUT_N, freq=FREQ),
    }


def run_deepar_longterm(
    seg: str,
    series_obs: pd.DataFrame,
    final_feature_df: pd.DataFrame,
    feature_cols: pd.Index,
    obs_end: pd.Timestamp,
    prediction_end_date: pd.Timestamp,
) -> Dict:
    """Train DeepAR on the full observed history and forecast out to `prediction_end_date`."""
    train_target = series_obs[seg].values
    train_dates = series_obs[DATE_COL]

    future_start = (obs_end + pd.DateOffset(months=1)) + pd.offsets.MonthEnd(0)
    future_dates = pd.date_range(start=future_start, end=prediction_end_date, freq=FREQ)

    full_dates = pd.concat(
        [train_dates.reset_index(drop=True), pd.Series(future_dates)], ignore_index=True
    )

    long_feature_df = pd.DataFrame({DATE_COL: full_dates}).merge(
        final_feature_df, on=DATE_COL, how="left"
    )
    long_feature_df[feature_cols] = long_feature_df[feature_cols].ffill().fillna(0)
    long_feature_df.replace([np.inf, -np.inf], 0, inplace=True)

    macro_feats = long_feature_df[MACRO_COLS_FOR_DEEPAR].T.values
    time_feats = create_time_features(full_dates)
    feat_dynamic_real = np.concatenate([macro_feats, time_feats], axis=0)

    train_ds = ListDataset(
        [{
            "start": train_dates.iloc[0],
            "target": train_target,
            "feat_dynamic_real": feat_dynamic_real[:, :len(train_target)],
        }],
        freq=FREQ,
    )
    test_ds = ListDataset(
        [{
            "start": train_dates.iloc[0],
            "target": train_target,
            "feat_dynamic_real": feat_dynamic_real,
        }],
        freq=FREQ,
    )

    estimator = DeepAREstimator(
        freq=FREQ,
        prediction_length=len(future_dates),
        context_length=min(len(train_target), FORECAST_STEP * 2),
        trainer_kwargs={
            "max_epochs": 30, "logger": False, "enable_progress_bar": False,
            "accelerator": "cpu",
        },
    )
    predictor = estimator.train(train_ds)
    forecast = next(iter(predictor.predict(test_ds)))

    return {
        "preds_mean": forecast.mean,
        "preds_low": forecast.quantile(0.10),
        "preds_up": forecast.quantile(0.90),
        "future_dates": future_dates,
    }
