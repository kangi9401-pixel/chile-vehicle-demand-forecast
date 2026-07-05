"""End-to-end orchestration: load data -> per-segment DeepAR (holdout + long-term) ->
XGBoost holdout -> ensemble evaluation -> plots + Excel summary."""

import logging

import numpy as np
import pandas as pd

from chile_forecast.config import (
    DATA_PATH, DATE_COL, HOLDOUT_EXCEL_PATH, HOLDOUT_N, MIN_SERIES_LEN,
    OBS_END, OUTPUT_DIR, PREDICTION_END_DATE, SEGMENT_COLS,
)
from chile_forecast.deepar_model import run_deepar_holdout, run_deepar_longterm
from chile_forecast.ensemble import build_holdout_row, print_anac_fit
from chile_forecast.features import build_feature_dataframe, get_longest_non_nan_slice
from chile_forecast.visualize import plot_observed_and_forecast
from chile_forecast.xgb_model import run_xgb_holdout

logger = logging.getLogger(__name__)


def load_raw_data() -> pd.DataFrame:
    raw_df = pd.read_csv(DATA_PATH)
    raw_df[DATE_COL] = pd.to_datetime(raw_df[DATE_COL].astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    raw_df.sort_values(DATE_COL, inplace=True)
    raw_df.reset_index(drop=True, inplace=True)
    return raw_df


def run() -> pd.DataFrame:
    raw_df = load_raw_data()
    final_feature_df = build_feature_dataframe(raw_df)
    feature_cols = final_feature_df.columns.drop(DATE_COL)

    raw_df_obs = raw_df[raw_df[DATE_COL] <= OBS_END].copy()
    final_feature_df_obs = final_feature_df[final_feature_df[DATE_COL] <= OBS_END].copy()

    holdout_end = raw_df_obs[DATE_COL].max()
    holdout_start = holdout_end - pd.DateOffset(months=HOLDOUT_N - 1)

    deepar_holdout_results = {}
    deepar_longterm_results = {}
    series_obs_by_segment = {}

    for seg in SEGMENT_COLS:
        logger.info("=== Processing: %s ===", seg)

        s_full = raw_df[seg]
        start_idx, end_idx, usable_len = get_longest_non_nan_slice(s_full)
        if usable_len < MIN_SERIES_LEN:
            logger.warning("%s: insufficient data (%d < %d), skipping", seg, usable_len, MIN_SERIES_LEN)
            continue

        series_full = pd.DataFrame({
            DATE_COL: raw_df[DATE_COL].iloc[start_idx:end_idx].values,
            seg: s_full.iloc[start_idx:end_idx].values,
        })
        series_obs = series_full[series_full[DATE_COL] <= OBS_END].reset_index(drop=True)
        if len(series_obs) < MIN_SERIES_LEN:
            logger.warning("%s: insufficient observed data, skipping", seg)
            continue

        series_obs_by_segment[seg] = series_obs

        deepar_holdout_results[seg] = run_deepar_holdout(
            seg, series_obs, final_feature_df_obs, holdout_start
        )
        logger.info("[%s] holdout DeepAR done", seg)

        deepar_longterm_results[seg] = run_deepar_longterm(
            seg, series_obs, final_feature_df, feature_cols, OBS_END, PREDICTION_END_DATE
        )
        logger.info("[%s] long-term DeepAR done", seg)

        plot_path = plot_observed_and_forecast(seg, series_obs, deepar_longterm_results[seg], OUTPUT_DIR)
        logger.info("[%s] forecast plot saved to %s", seg, plot_path)

    logger.info("=== Final Holdout Evaluation ===")
    train_mask = final_feature_df[DATE_COL] < holdout_start
    test_mask = (final_feature_df[DATE_COL] >= holdout_start) & (final_feature_df[DATE_COL] <= holdout_end)

    X_train_base = final_feature_df.loc[train_mask].drop(columns=[DATE_COL])
    X_test_base = final_feature_df.loc[test_mask].drop(columns=[DATE_COL])

    holdout_summary = []
    for seg in SEGMENT_COLS:
        if seg not in series_obs_by_segment:
            continue

        y_train = raw_df.loc[train_mask, seg].ffill().fillna(0).values
        y_test = raw_df.loc[test_mask, seg].values
        if len(y_train) < MIN_SERIES_LEN or len(y_test) == 0:
            logger.warning("%s: insufficient holdout data, skipping", seg)
            continue

        preds_deepar = deepar_holdout_results.get(seg, {}).get(
            "preds_mean", np.full(HOLDOUT_N, np.nan)
        )
        preds_xgb = run_xgb_holdout(X_train_base, y_train, X_test_base)

        holdout_summary.append(build_holdout_row(seg, y_test, preds_deepar, preds_xgb, holdout_start))

    holdout_df = pd.DataFrame(holdout_summary)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    holdout_df.to_excel(HOLDOUT_EXCEL_PATH, index=False)
    logger.info("Holdout results saved to %s", HOLDOUT_EXCEL_PATH)

    if not holdout_df.empty:
        logger.info(
            "\n%s",
            holdout_df[["Segment", "MAE_DeepAR", "MAE_XGBoost", "MAE_Ensemble"]].round(2),
        )

    print_anac_fit(holdout_df)
    return holdout_df
