"""DeepAR training utilities.

`train_and_forecast` is the single low-level trainer used by the rolling-origin
evaluation (`evaluation.py`) and by the legacy wrappers in
`legacy/deepar_windows.py`.
"""

import logging
import tempfile
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from gluonts.dataset.common import ListDataset
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.torch.model.deepar import DeepAREstimator

from chile_forecast.config import DATE_COL, FORECAST_STEP, FREQ, MACRO_COLS_FOR_DEEPAR, RANDOM_SEED
from chile_forecast.probabilistic import QUANTILE_LEVELS

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

# GluonTS 0.16 emits this PyTorch 2.9 migration warning once per training batch;
# it is not actionable for this pinned environment and otherwise obscures results.
warnings.filterwarnings(
    "ignore",
    message="Using a non-tuple sequence for multidimensional indexing.*",
    category=UserWarning,
)


def create_time_features(dates: pd.Series, freq: str = FREQ) -> np.ndarray:
    feats = time_features_from_frequency_str(freq)
    return np.stack([f(pd.DatetimeIndex(dates)) for f in feats], axis=0)


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
    # All stochastic libraries used by GluonTS/Lightning are fixed at the same seed.
    import random
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

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

    with tempfile.TemporaryDirectory(prefix="chile_forecast_deepar_") as checkpoint_dir:
        estimator = DeepAREstimator(
            freq=FREQ,
            prediction_length=prediction_length,
            context_length=min(len(train_target), FORECAST_STEP * 2),
            trainer_kwargs={
                "max_epochs": max_epochs, "logger": False, "enable_progress_bar": False,
                "accelerator": "cpu", "deterministic": True, "enable_model_summary": False,
                "default_root_dir": checkpoint_dir,
            },
            **deepar_kwargs,
        )
        predictor = estimator.train(train_ds)
        forecast = next(iter(predictor.predict(test_ds)))

    return {
        "preds_mean": forecast.mean,
        "preds_low": forecast.quantile(0.10),
        "preds_up": forecast.quantile(0.90),
        # Read from the samples already drawn above, so this consumes no randomness.
        "quantiles": np.column_stack([forecast.quantile(level) for level in QUANTILE_LEVELS]),
    }

