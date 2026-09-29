from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chile_forecast.baselines import seasonal_naive
from chile_forecast.config import BACKTEST_HORIZONS, DATE_COL
from chile_forecast.evaluation import common_origins, evaluate_target
from chile_forecast.metrics import calc_fit_metrics
from chile_forecast.optimization import run_strategy_comparison
from chile_forecast.pipeline import load_raw_data, run
from chile_forecast.preprocessing import LeakageSafePreprocessor
from chile_forecast.promotion_response import build_synthetic_promotion_inputs


def test_future_rows_cannot_change_training_preprocessing():
    raw = load_raw_data().iloc[:80].copy()
    train = raw.iloc[:60].copy()
    future_a = raw.iloc[60:].copy()
    future_b = future_a.copy()
    future_b["Copper_USD_per_Ton"] *= 1_000_000
    future_b["Interest Rate"] = -999

    pre_a = LeakageSafePreprocessor().fit(train)
    pre_b = LeakageSafePreprocessor().fit(pd.concat([train, future_b]).iloc[:60])
    pd.testing.assert_frame_equal(pre_a.transform(train), pre_b.transform(train))
    assert pre_a.resource_means_ == pre_b.resource_means_
    # Future feature construction consults dates and the last training row only.
    pd.testing.assert_frame_equal(
        pre_a.make_future_features(train, future_a[DATE_COL]),
        pre_b.make_future_features(train, future_b[DATE_COL]),
    )


def test_unstable_resource_price_denominator_features_are_absent():
    raw = load_raw_data().iloc[:60]
    features = LeakageSafePreprocessor().fit_transform(raw)
    assert not any("/avg_resource_price" in col for col in features.columns)
    assert np.isfinite(features.drop(columns=DATE_COL).to_numpy()).all()


def test_seasonal_naive_is_exactly_previous_year_month():
    train = np.arange(1.0, 25.0)
    expected = np.arange(13.0, 25.0)
    np.testing.assert_array_equal(seasonal_naive(train, 12), expected)


def test_wape_and_smape_handle_zero_actuals():
    metrics = calc_fit_metrics([0, 100], [20, 80])
    assert metrics["MAE"] == pytest.approx(20)
    assert metrics["WAPE"] == pytest.approx(40)
    assert np.isfinite(metrics["sMAPE"])


def test_models_share_origins_horizons_and_exact_target_dates():
    raw = load_raw_data()
    predictions, metrics = evaluate_target(
        raw,
        "B-Sedan",
        horizons=(3, 6),
        n_origins=2,
        include_deepar=False,
    )
    expected_models = {"LastValue", "SeasonalNaive", "MovingAverage6", "XGBoost"}
    assert set(metrics["model"]) == expected_models
    counts = metrics.groupby(["origin_number", "horizon"])["model"].nunique()
    assert (counts == len(expected_models)).all()

    for (_, horizon, model), group in predictions.groupby(["origin_number", "horizon", "model"]):
        assert len(group) == horizon
        assert group["lead"].tolist() == list(range(1, horizon + 1))
        assert group["date"].is_monotonic_increasing
        assert group["actual"].notna().all()


def test_common_origin_design_has_eight_full_horizon_cutoffs():
    raw = load_raw_data()
    origins = common_origins(raw[DATE_COL], max(BACKTEST_HORIZONS), n_origins=8)
    assert len(origins) == 8
    assert origins == sorted(origins)
    assert all((b.to_period("M") - a.to_period("M")).n == 3 for a, b in zip(origins, origins[1:]))


def _fake_forecast_outputs():
    targets = ["B_HB", "B-Sedan", "SUV-A", "SUV-B"]
    predictions = []
    summary = []
    for idx, target in enumerate(targets):
        for lead in range(1, 4):
            predictions.append({
                "target": target,
                "origin_number": 8,
                "horizon": 3,
                "lead": lead,
                "model": "SeasonalNaive",
                "predicted": 900 + idx * 100 + lead,
            })
        summary.append({"target": target, "horizon": 3, "model": "SeasonalNaive", "WAPE_mean": 10 + idx})
    return pd.DataFrame(predictions), pd.DataFrame(summary)


def test_optimization_respects_all_budget_supply_factory_and_regulatory_constraints():
    predictions, summary = _fake_forecast_outputs()
    models, response = build_synthetic_promotion_inputs(predictions, summary)
    detail, comparison, checks, scenarios = run_strategy_comparison(models, response)

    assert checks["all_constraints_ok"].all()
    assert (comparison["total_budget_mclp"] <= 300 + 1e-6).all()
    assert (detail["allocated_mclp"] >= detail["min_budget_mclp"] - 1e-6).all()
    assert (detail["allocated_mclp"] <= detail["max_budget_mclp"] + 1e-6).all()
    assert (detail["expected_sales"] <= detail["inventory"] + detail["supply_available"] + 1e-6).all()
    assert (detail.loc[~detail["regulatory_allowed"], "allocated_mclp"] == 0).all()
    assert set(scenarios["scenario"]) == {"downside", "base", "upside"}


def test_fixed_seed_makes_non_deep_evaluation_reproducible():
    raw = load_raw_data()
    first_pred, first_metrics = evaluate_target(raw, "SUV-B", horizons=(3,), n_origins=2, include_deepar=False)
    second_pred, second_metrics = evaluate_target(raw, "SUV-B", horizons=(3,), n_origins=2, include_deepar=False)
    pd.testing.assert_frame_equal(first_pred, second_pred)
    pd.testing.assert_frame_equal(first_metrics, second_metrics)


def test_fixed_seed_makes_deepar_prediction_reproducible():
    raw = load_raw_data()
    first, _ = evaluate_target(raw, "SUV-B", horizons=(3,), n_origins=1, include_deepar=True, deepar_epochs=1)
    second, _ = evaluate_target(raw, "SUV-B", horizons=(3,), n_origins=1, include_deepar=True, deepar_epochs=1)
    first_values = first.loc[first["model"] == "DeepAR", "predicted"].to_numpy()
    second_values = second.loc[second["model"] == "DeepAR", "predicted"].to_numpy()
    np.testing.assert_array_equal(first_values, second_values)


def test_full_synthetic_pipeline_smoke(tmp_path):
    comparison = run(include_deepar=False, output_dir=tmp_path)
    assert set(comparison["strategy"]) == {"Equal", "ForecastShare", "PriorYearShare", "Optimized"}
    required = {
        "forecast_predictions.csv",
        "forecast_metrics_by_origin.csv",
        "forecast_metrics_summary.csv",
        "allocation_strategy_comparison.csv",
        "constraint_checks.csv",
        "scenario_summary.csv",
        "forecast_actual_vs_predicted.png",
        "scenario_budget_allocations.png",
    }
    assert required <= {path.name for path in tmp_path.iterdir()}
