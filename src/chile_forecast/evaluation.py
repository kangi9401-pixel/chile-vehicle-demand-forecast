"""Common-origin, multi-horizon forecasting evaluation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import pandas as pd

from chile_forecast.baselines import baseline_predictions
from chile_forecast.config import (
    BACKTEST_HORIZONS,
    BACKTEST_ORIGINS,
    BACKTEST_STEP_MONTHS,
    DATE_COL,
    DEEPAR_BACKTEST_EPOCHS,
    RANDOM_SEED,
)
from chile_forecast.deepar_model import build_feat_dynamic_real, train_and_forecast
from chile_forecast.metrics import calc_fit_metrics, seasonal_mase_scale
from chile_forecast.preprocessing import LeakageSafePreprocessor
from chile_forecast.xgb_model import run_xgb_holdout

logger = logging.getLogger(__name__)


def common_origins(
    dates: Iterable[pd.Timestamp],
    max_horizon: int = 12,
    n_origins: int = BACKTEST_ORIGINS,
    step_months: int = BACKTEST_STEP_MONTHS,
) -> list[pd.Timestamp]:
    """Return ascending training cutoffs with a full max-horizon test window."""
    unique = pd.DatetimeIndex(pd.Series(dates).dropna().unique()).sort_values()
    if len(unique) <= max_horizon:
        raise ValueError("not enough observations for requested horizon")
    latest_cutoff = unique[-max_horizon - 1]
    cutoffs = [
        latest_cutoff - pd.DateOffset(months=step_months * i) + pd.offsets.MonthEnd(0)
        for i in range(n_origins)
    ]
    available = set(unique)
    cutoffs = [pd.Timestamp(c) for c in cutoffs if pd.Timestamp(c) in available]
    if len(cutoffs) != n_origins:
        raise ValueError(f"requested {n_origins} origins but only {len(cutoffs)} are available")
    return sorted(cutoffs)


def _inverse_mae_weight(history: list[Tuple[float, float]]) -> float:
    if not history:
        return 0.5
    d_mae = float(np.mean([row[0] for row in history]))
    x_mae = float(np.mean([row[1] for row in history]))
    if not np.isfinite(d_mae + x_mae) or d_mae + x_mae <= 0:
        return 0.5
    return float(x_mae / (d_mae + x_mae))


ErrorRecord = Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]


def _visible_history(records: Sequence[ErrorRecord], cutoff: pd.Timestamp) -> list[Tuple[float, float]]:
    """Per-origin (DeepAR MAE, XGBoost MAE) over target dates observed by `cutoff`.

    Backtest windows overlap, so an earlier origin's forecast window can extend past
    the current cutoff; only its errors on dates <= `cutoff` are visible.  Origins
    with no visible dates are dropped.
    """
    cutoff = pd.Timestamp(cutoff)
    history = []
    for dates, deepar_abs_error, xgb_abs_error in records:
        visible = np.asarray(pd.DatetimeIndex(dates) <= cutoff)
        if visible.any():
            history.append((
                float(np.mean(np.asarray(deepar_abs_error)[visible])),
                float(np.mean(np.asarray(xgb_abs_error)[visible])),
            ))
    return history


def evaluate_target(
    raw_df: pd.DataFrame,
    target: str,
    horizons: Sequence[int] = BACKTEST_HORIZONS,
    n_origins: int = BACKTEST_ORIGINS,
    step_months: int = BACKTEST_STEP_MONTHS,
    include_deepar: bool = True,
    deepar_epochs: int = DEEPAR_BACKTEST_EPOCHS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate every model on identical cutoffs and target dates.

    Hyperparameters are fixed in advance.  The ensemble weight at an origin uses only
    earlier origins' errors on target dates already observed by the current cutoff,
    so no actual value after the cutoff influences the weight.
    """
    if target not in raw_df:
        raise ValueError(f"unknown target: {target}")
    series = raw_df.loc[raw_df[target].notna()].copy().reset_index(drop=True)
    max_horizon = max(horizons)
    origins = common_origins(series[DATE_COL], max_horizon, n_origins, step_months)
    prediction_rows = []
    metric_rows = []
    error_records: list[ErrorRecord] = []

    for origin_number, cutoff in enumerate(origins, start=1):
        train = series.loc[series[DATE_COL] <= cutoff].copy()
        test = series.loc[series[DATE_COL] > cutoff].head(max_horizon).copy()
        if len(test) != max_horizon:
            raise AssertionError("origin does not have a complete test window")

        preprocessor = LeakageSafePreprocessor().fit(train)
        train_features = preprocessor.transform(train)
        future_features = preprocessor.make_future_features(train, test[DATE_COL])
        X_train = train_features.drop(columns=DATE_COL)
        X_future = future_features.drop(columns=DATE_COL)
        y_train = train[target].to_numpy(dtype=float)
        y_actual = test[target].to_numpy(dtype=float)

        all_predictions: Dict[str, np.ndarray] = baseline_predictions(y_train, max_horizon)
        all_predictions["XGBoost"] = run_xgb_holdout(
            X_train,
            y_train,
            X_future,
            xgb_params={"n_estimators": 150, "n_jobs": 1, "random_state": RANDOM_SEED},
        )

        ensemble_weight = np.nan
        if include_deepar:
            safe_feature_source = pd.concat([train_features, future_features], ignore_index=True)
            all_dates = pd.concat([train[DATE_COL], test[DATE_COL]], ignore_index=True)
            dynamic = build_feat_dynamic_real(all_dates, safe_feature_source)
            deep = train_and_forecast(
                train[DATE_COL],
                y_train,
                dynamic,
                prediction_length=max_horizon,
                max_epochs=deepar_epochs,
                deepar_kwargs={"hidden_size": 24, "num_layers": 2, "lr": 1e-3},
            )
            all_predictions["DeepAR"] = np.asarray(deep["preds_mean"], dtype=float)
            ensemble_weight = _inverse_mae_weight(_visible_history(error_records, cutoff))
            all_predictions["LeakageSafeEnsemble"] = (
                ensemble_weight * all_predictions["DeepAR"]
                + (1 - ensemble_weight) * all_predictions["XGBoost"]
            )

        mase_scale = seasonal_mase_scale(y_train)
        for horizon in horizons:
            actual_h = y_actual[:horizon]
            dates_h = test[DATE_COL].iloc[:horizon]
            for model, predictions in all_predictions.items():
                pred_h = np.maximum(np.asarray(predictions[:horizon], dtype=float), 0.0)
                metrics = calc_fit_metrics(actual_h, pred_h, mase_scale=mase_scale)
                metric_rows.append({
                    "target": target,
                    "origin_number": origin_number,
                    "train_end": cutoff,
                    "horizon": horizon,
                    "model": model,
                    "ensemble_weight_deepar": ensemble_weight if model == "LeakageSafeEnsemble" else np.nan,
                    **{key: metrics[key] for key in ("MAE", "RMSE", "WAPE", "sMAPE", "MASE")},
                })
                for lead, (date, actual, predicted) in enumerate(zip(dates_h, actual_h, pred_h), start=1):
                    prediction_rows.append({
                        "target": target,
                        "origin_number": origin_number,
                        "train_end": cutoff,
                        "horizon": horizon,
                        "lead": lead,
                        "date": date,
                        "model": model,
                        "actual": float(actual),
                        "predicted": float(predicted),
                    })

        if include_deepar:
            # Errors use the raw (unclipped) predictions, as the weight always has.
            error_records.append((
                pd.DatetimeIndex(test[DATE_COL]),
                np.abs(y_actual - all_predictions["DeepAR"]),
                np.abs(y_actual - all_predictions["XGBoost"]),
            ))
        logger.info("%s origin %d/%d complete (%s)", target, origin_number, len(origins), cutoff.date())

    return pd.DataFrame(prediction_rows), pd.DataFrame(metric_rows)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    measure_cols = ["MAE", "RMSE", "WAPE", "sMAPE", "MASE"]
    summary = (
        metrics.groupby(["target", "horizon", "model"], as_index=False)[measure_cols]
        .agg(["mean", "std"])
    )
    summary.columns = ["_".join(c).rstrip("_") for c in summary.columns.to_flat_index()]
    return summary


def seasonal_improvement(summary: pd.DataFrame) -> pd.DataFrame:
    base = summary.loc[summary["model"] == "SeasonalNaive", ["target", "horizon", "WAPE_mean"]]
    base = base.rename(columns={"WAPE_mean": "seasonal_naive_WAPE"})
    result = summary.merge(base, on=["target", "horizon"], how="left")
    result["WAPE_improvement_vs_seasonal_pct"] = (
        (result["seasonal_naive_WAPE"] - result["WAPE_mean"])
        / result["seasonal_naive_WAPE"].replace(0, np.nan)
        * 100
    )
    return result


def save_evaluation(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    output_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_metrics(metrics)
    improvement = seasonal_improvement(summary)
    predictions.to_csv(output_dir / "forecast_predictions.csv", index=False)
    metrics.to_csv(output_dir / "forecast_metrics_by_origin.csv", index=False)
    summary.to_csv(output_dir / "forecast_metrics_summary.csv", index=False)
    improvement.to_csv(output_dir / "forecast_improvement_vs_seasonal.csv", index=False)
    return summary, improvement
