"""End-to-end orchestration.

``run`` is the default leakage-safe forecast-to-optimization pipeline.  The former
3-month forecast workflow remains available as ``run_legacy`` solely so earlier
results can be audited and reproduced; it is not used for final reported metrics.
"""

import logging

import pandas as pd

from chile_forecast.backtest import rolling_origin_backtest
from chile_forecast.config import (
    DATA_PATH, DATE_COL, HOLDOUT_EXCEL_PATH, HOLDOUT_N, MIN_SERIES_LEN,
    OBS_END, OUTPUT_DIR, PREDICTION_END_DATE, SEGMENT_COLS, TUNING_SEGMENT, VAL_N,
)
from chile_forecast.deepar_model import run_deepar_holdout, run_deepar_longterm
from chile_forecast.ensemble import build_holdout_row, compute_ensemble_weights, print_anac_fit
from chile_forecast.features import build_feature_dataframe, get_longest_non_nan_slice
from chile_forecast.interpret import shap_summary_plot
from chile_forecast.tuning import tune_deepar_hyperparameters, tune_xgb_hyperparameters
from chile_forecast.visualize import plot_observed_and_forecast
from chile_forecast.xgb_model import fit_xgb, run_xgb_holdout

logger = logging.getLogger(__name__)


def load_raw_data() -> pd.DataFrame:
    raw_df = pd.read_csv(DATA_PATH)
    raw_df[DATE_COL] = pd.to_datetime(raw_df[DATE_COL].astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    raw_df.sort_values(DATE_COL, inplace=True)
    raw_df.reset_index(drop=True, inplace=True)
    return raw_df


def _segment_series(seg: str, raw_df: pd.DataFrame) -> pd.DataFrame:
    """Longest usable contiguous history for `seg`, clipped to the observed window."""
    s_full = raw_df[seg]
    start_idx, end_idx, usable_len = get_longest_non_nan_slice(s_full)
    if usable_len < MIN_SERIES_LEN:
        return pd.DataFrame()

    series_full = pd.DataFrame({
        DATE_COL: raw_df[DATE_COL].iloc[start_idx:end_idx].values,
        seg: s_full.iloc[start_idx:end_idx].values,
    })
    series_obs = series_full[series_full[DATE_COL] <= OBS_END].reset_index(drop=True)
    return series_obs if len(series_obs) >= MIN_SERIES_LEN else pd.DataFrame()


def run_legacy() -> pd.DataFrame:
    raw_df = load_raw_data()
    final_feature_df = build_feature_dataframe(raw_df)
    feature_cols = final_feature_df.columns.drop(DATE_COL)

    raw_df_obs = raw_df[raw_df[DATE_COL] <= OBS_END].copy()
    final_feature_df_obs = final_feature_df[final_feature_df[DATE_COL] <= OBS_END].copy()

    holdout_end = raw_df_obs[DATE_COL].max()
    holdout_start = holdout_end - pd.DateOffset(months=HOLDOUT_N - 1)
    train_mask = final_feature_df[DATE_COL] < holdout_start
    test_mask = (final_feature_df[DATE_COL] >= holdout_start) & (final_feature_df[DATE_COL] <= holdout_end)
    X_train_base = final_feature_df.loc[train_mask].drop(columns=[DATE_COL])
    X_test_base = final_feature_df.loc[test_mask].drop(columns=[DATE_COL])

    series_obs_by_segment = {seg: _segment_series(seg, raw_df) for seg in SEGMENT_COLS}
    series_obs_by_segment = {k: v for k, v in series_obs_by_segment.items() if not v.empty}
    if TUNING_SEGMENT not in series_obs_by_segment:
        raise ValueError(f"TUNING_SEGMENT '{TUNING_SEGMENT}' has insufficient data to tune on")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Hyperparameter search (once, on TUNING_SEGMENT) ---------------------------
    logger.info("=== Hyperparameter search (segment: %s) ===", TUNING_SEGMENT)
    y_train_tuning = raw_df.loc[train_mask, TUNING_SEGMENT].ffill().fillna(0).values
    xgb_params, xgb_search_results = tune_xgb_hyperparameters(X_train_base, y_train_tuning)
    logger.info("Best XGBoost params: %s", xgb_params)
    xgb_search_results.to_csv(OUTPUT_DIR / "xgb_tuning_results.csv", index=False)

    deepar_kwargs, deepar_search_results = tune_deepar_hyperparameters(
        TUNING_SEGMENT, series_obs_by_segment[TUNING_SEGMENT], final_feature_df_obs, VAL_N, HOLDOUT_N,
    )
    logger.info("Best DeepAR config: %s", deepar_kwargs)
    deepar_search_results.to_csv(OUTPUT_DIR / "deepar_tuning_results.csv", index=False)

    # --- Per-segment rolling backtest, holdout, long-term forecast, SHAP -----------
    rolling_frames = []
    holdout_summary = []

    for seg, series_obs in series_obs_by_segment.items():
        logger.info("=== Processing: %s ===", seg)

        rolling_df = rolling_origin_backtest(
            seg, series_obs, final_feature_df, final_feature_df_obs, raw_df,
            deepar_kwargs=deepar_kwargs, xgb_params=xgb_params,
        )
        rolling_frames.append(rolling_df)
        logger.info("[%s] rolling-origin backtest done (%d origins)", seg, len(rolling_df))

        deepar_holdout_result = run_deepar_holdout(
            seg, series_obs, final_feature_df_obs, holdout_start, deepar_kwargs=deepar_kwargs,
        )
        logger.info("[%s] holdout DeepAR done", seg)

        y_train = raw_df.loc[train_mask, seg].ffill().fillna(0).values
        y_test = raw_df.loc[test_mask, seg].values
        preds_xgb = run_xgb_holdout(X_train_base, y_train, X_test_base, xgb_params=xgb_params)

        weight_deepar = compute_ensemble_weights(rolling_df).get(seg, 0.5)
        holdout_summary.append(build_holdout_row(
            seg, y_test, deepar_holdout_result["preds_mean"], preds_xgb, holdout_start, weight_deepar,
        ))

        deepar_longterm_result = run_deepar_longterm(
            seg, series_obs, final_feature_df, feature_cols, OBS_END, PREDICTION_END_DATE,
            deepar_kwargs=deepar_kwargs,
        )
        logger.info("[%s] long-term DeepAR done", seg)

        plot_path = plot_observed_and_forecast(seg, series_obs, deepar_longterm_result, OUTPUT_DIR)
        logger.info("[%s] forecast plot saved to %s", seg, plot_path)

        xgb_model = fit_xgb(X_train_base, y_train, xgb_params=xgb_params)
        shap_summary_plot(xgb_model, X_train_base, seg, OUTPUT_DIR)

    # --- Write results ---------------------------------------------------------
    rolling_all = pd.concat(rolling_frames, ignore_index=True) if rolling_frames else pd.DataFrame()
    rolling_all.to_excel(OUTPUT_DIR / "rolling_origin_backtest.xlsx", index=False)
    if not rolling_all.empty:
        origins_per_segment = rolling_all.groupby("Segment").size()
        agg = rolling_all.groupby("Segment")[["MAE_DeepAR", "MAE_XGBoost", "MAE_Ensemble_5050"]].agg(["mean", "std"])
        logger.info("=== Rolling-origin backtest (mean +/- std across up to %d origins) ===", origins_per_segment.max())
        logger.info("\n%s", agg.round(2))

    holdout_df = pd.DataFrame(holdout_summary)
    holdout_df.to_excel(HOLDOUT_EXCEL_PATH, index=False)
    logger.info("Holdout results saved to %s", HOLDOUT_EXCEL_PATH)

    if not holdout_df.empty:
        logger.info(
            "\n%s",
            holdout_df[["Segment", "Weight_DeepAR", "MAE_DeepAR", "MAE_XGBoost", "MAE_Ensemble"]].round(3),
        )

    print_anac_fit(holdout_df)
    return holdout_df


def run(
    include_deepar: bool = True,
    deepar_epochs: int = None,
    output_dir=None,
) -> pd.DataFrame:
    """Run the leakage-safe forecasting and promotion decision pipeline."""
    from pathlib import Path

    from chile_forecast.config import (
        BACKTEST_TARGETS, DECISION_OUTPUT_DIR, DEEPAR_BACKTEST_EPOCHS,
    )
    from chile_forecast.decision_visualization import create_decision_charts
    from chile_forecast.evaluation import evaluate_target, save_evaluation
    from chile_forecast.optimization import run_strategy_comparison
    from chile_forecast.promotion_response import build_synthetic_promotion_inputs

    destination = DECISION_OUTPUT_DIR if output_dir is None else Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    epochs = DEEPAR_BACKTEST_EPOCHS if deepar_epochs is None else int(deepar_epochs)
    raw_df = load_raw_data()

    prediction_frames, metric_frames = [], []
    for target in BACKTEST_TARGETS:
        predictions, metrics = evaluate_target(
            raw_df,
            target,
            include_deepar=include_deepar,
            deepar_epochs=epochs,
        )
        prediction_frames.append(predictions)
        metric_frames.append(metrics)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = pd.concat(metric_frames, ignore_index=True)
    summary, _ = save_evaluation(predictions, metrics, destination)

    models, response = build_synthetic_promotion_inputs(predictions, summary)
    models.to_csv(destination / "synthetic_promotion_inputs.csv", index=False)
    response.to_csv(destination / "synthetic_promotion_response.csv", index=False)
    detail, comparison, checks, scenarios = run_strategy_comparison(models, response)
    detail.to_csv(destination / "allocation_detail.csv", index=False)
    comparison.to_csv(destination / "allocation_strategy_comparison.csv", index=False)
    checks.to_csv(destination / "constraint_checks.csv", index=False)
    scenarios.to_csv(destination / "scenario_allocation_detail.csv", index=False)
    scenario_summary = scenarios.groupby("scenario", as_index=False).agg(
        total_budget_mclp=("allocated_mclp", "sum"),
        expected_sales=("expected_sales", "sum"),
        expected_incremental_sales=("expected_incremental_sales", "sum"),
        expected_revenue_mclp=("expected_revenue_mclp", "sum"),
        expected_contribution_mclp=("expected_contribution_mclp", "sum"),
        expected_incremental_profit_mclp=("expected_incremental_profit_mclp", "sum"),
    )
    scenario_summary.to_csv(destination / "scenario_summary.csv", index=False)
    create_decision_charts(predictions, metrics, summary, detail, comparison, scenarios, destination)
    logger.info("Decision pipeline outputs saved to %s", destination)
    return comparison
