"""End-to-end orchestration of the leakage-safe forecast-to-optimization pipeline.

The former 3-month forecast workflow lives in ``chile_forecast.legacy.pipeline``.
"""

import logging

import pandas as pd

from chile_forecast.config import DATA_PATH, DATE_COL

logger = logging.getLogger(__name__)


def load_raw_data() -> pd.DataFrame:
    raw_df = pd.read_csv(DATA_PATH)
    raw_df[DATE_COL] = pd.to_datetime(raw_df[DATE_COL].astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    raw_df.sort_values(DATE_COL, inplace=True)
    raw_df.reset_index(drop=True, inplace=True)
    return raw_df


def run(
    include_deepar: bool = True,
    deepar_epochs: int = None,
    output_dir=None,
) -> pd.DataFrame:
    """Run the leakage-safe forecasting and promotion decision pipeline."""
    from pathlib import Path

    from chile_forecast.config import (
        BACKTEST_HORIZONS, BACKTEST_TARGETS, DECISION_OUTPUT_DIR, DEEPAR_BACKTEST_EPOCHS,
    )
    from chile_forecast.decision_visualization import create_decision_charts, create_extension_charts
    from chile_forecast.evaluation import evaluate_target, save_evaluation
    from chile_forecast.optimization import run_response_sensitivity, run_strategy_comparison
    from chile_forecast.probabilistic import quantile_calibration, summarize_probabilistic
    from chile_forecast.promotion_response import build_synthetic_promotion_inputs
    from chile_forecast.significance import forecast_significance

    destination = DECISION_OUTPUT_DIR if output_dir is None else Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    epochs = DEEPAR_BACKTEST_EPOCHS if deepar_epochs is None else int(deepar_epochs)
    raw_df = load_raw_data()

    prediction_frames, metric_frames, quantile_frames = [], [], []
    for target in BACKTEST_TARGETS:
        predictions, metrics, quantiles = evaluate_target(
            raw_df,
            target,
            include_deepar=include_deepar,
            deepar_epochs=epochs,
            return_quantiles=True,
        )
        prediction_frames.append(predictions)
        metric_frames.append(metrics)
        quantile_frames.append(quantiles)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = pd.concat(metric_frames, ignore_index=True)
    quantiles = pd.concat(quantile_frames, ignore_index=True)
    summary, _ = save_evaluation(predictions, metrics, destination)

    quantiles.to_csv(destination / "forecast_quantiles.csv", index=False)
    summarize_probabilistic(quantiles, BACKTEST_HORIZONS).to_csv(
        destination / "probabilistic_metrics_summary.csv", index=False
    )
    calibration = quantile_calibration(quantiles, max(BACKTEST_HORIZONS))
    calibration.to_csv(destination / "quantile_calibration.csv", index=False)
    forecast_significance(metrics).to_csv(destination / "forecast_significance.csv", index=False)

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
    sensitivity, sensitivity_by_model = run_response_sensitivity(models, response)
    sensitivity.to_csv(destination / "optimization_sensitivity.csv", index=False)
    sensitivity_by_model.to_csv(destination / "optimization_sensitivity_by_model.csv", index=False)
    create_decision_charts(predictions, metrics, summary, detail, comparison, scenarios, destination)
    create_extension_charts(calibration, sensitivity, destination)
    logger.info("Decision pipeline outputs saved to %s", destination)
    return comparison
