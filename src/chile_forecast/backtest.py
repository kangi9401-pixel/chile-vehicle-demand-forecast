"""Rolling-origin (walk-forward) backtest.

A single 3-month holdout is a single noisy draw -- it can't distinguish "this model
generalizes" from "this model got lucky on these particular 3 months." Repeating the
holdout evaluation at several earlier origins gives a distribution of the accuracy
metrics (mean +/- std across origins) instead of one point estimate, and is also
what feeds the per-segment ensemble weights (see `ensemble.compute_ensemble_weights`).
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from chile_forecast.config import DATE_COL, HOLDOUT_N, MIN_SERIES_LEN
from chile_forecast.deepar_model import run_deepar_window
from chile_forecast.xgb_model import run_xgb_holdout

logger = logging.getLogger(__name__)

N_ORIGINS = 3
ORIGIN_STEP_MONTHS = 3


def rolling_origin_backtest(
    seg: str,
    series_obs: pd.DataFrame,
    final_feature_df: pd.DataFrame,
    final_feature_df_obs: pd.DataFrame,
    raw_df: pd.DataFrame,
    n_origins: int = N_ORIGINS,
    step_months: int = ORIGIN_STEP_MONTHS,
    deepar_kwargs: Optional[Dict] = None,
    xgb_params: Optional[Dict] = None,
) -> pd.DataFrame:
    """Evaluate DeepAR/XGBoost/ensemble at `n_origins` holdout windows, each
    `step_months` further back than the last, all using the same HOLDOUT_N-month
    window length as the final evaluation. Origins with too little remaining
    training history for this segment are skipped."""
    last_date = series_obs[DATE_COL].iloc[-1]
    rows = []

    for i in range(n_origins):
        holdout_end = last_date - pd.DateOffset(months=step_months * i)
        train_end = holdout_end - pd.DateOffset(months=HOLDOUT_N)

        train_window = series_obs[series_obs[DATE_COL] <= train_end]
        test_window = series_obs[
            (series_obs[DATE_COL] > train_end) & (series_obs[DATE_COL] <= holdout_end)
        ]
        if len(train_window) < MIN_SERIES_LEN or len(test_window) < HOLDOUT_N:
            logger.warning("%s: origin %s skipped (insufficient history)", seg, holdout_end.date())
            continue

        deepar_result = run_deepar_window(
            seg, series_obs, final_feature_df_obs, train_end,
            prediction_length=HOLDOUT_N, max_epochs=20, deepar_kwargs=deepar_kwargs,
        )
        preds_deepar = deepar_result["preds_mean"]

        train_mask = final_feature_df[DATE_COL] <= train_end
        test_mask = (final_feature_df[DATE_COL] > train_end) & (final_feature_df[DATE_COL] <= holdout_end)
        X_train = final_feature_df.loc[train_mask].drop(columns=[DATE_COL])
        X_test = final_feature_df.loc[test_mask].drop(columns=[DATE_COL])
        y_train = raw_df.loc[train_mask, seg].ffill().fillna(0).values
        y_test = raw_df.loc[test_mask, seg].values
        preds_xgb = run_xgb_holdout(X_train, y_train, X_test, xgb_params=xgb_params)

        min_len = min(len(y_test), len(preds_deepar), len(preds_xgb))
        y_e = y_test[:min_len]
        d_e = np.asarray(preds_deepar)[:min_len]
        x_e = np.asarray(preds_xgb)[:min_len]
        ens_e = (d_e + x_e) / 2

        rows.append({
            "Segment": seg,
            "origin": holdout_end,
            "MAE_DeepAR": mean_absolute_error(y_e, d_e),
            "MAE_XGBoost": mean_absolute_error(y_e, x_e),
            "MAE_Ensemble_5050": mean_absolute_error(y_e, ens_e),
        })
        logger.info(
            "[%s] origin %s -> MAE DeepAR=%.1f XGB=%.1f",
            seg, holdout_end.date(), rows[-1]["MAE_DeepAR"], rows[-1]["MAE_XGBoost"],
        )

    return pd.DataFrame(rows)
