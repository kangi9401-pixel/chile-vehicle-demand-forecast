"""Probabilistic forecast evaluation: quantile forecasts, interval scores and CRPS.

DeepAR produces a predictive distribution; the point-forecast metrics only score its
mean.  This module scores the distribution itself against a simple probabilistic
baseline, a Seasonal Naive forecast widened by the empirical quantiles of its own
in-sample year-on-year errors.

CRPS is approximated from the quantile forecasts as twice the average pinball loss
over evenly spaced levels.  For a point forecast every quantile equals the point, and
this reduces exactly to the absolute error, so the scaled CRPS below is directly
comparable to WAPE.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

QUANTILE_LEVELS = tuple(round(0.05 * i, 2) for i in range(1, 20))  # 0.05 ... 0.95
INTERVAL_LOWER, INTERVAL_UPPER = 0.10, 0.90  # nominal 80% central interval


def quantile_column(level: float) -> str:
    return f"q{int(round(level * 100)):02d}"


def pinball_loss(actual, quantile, level: float) -> np.ndarray:
    diff = np.asarray(actual, dtype=float) - np.asarray(quantile, dtype=float)
    return np.maximum(level * diff, (level - 1.0) * diff)


def quantile_crps(actual, quantiles: np.ndarray, levels: Sequence[float] = QUANTILE_LEVELS) -> np.ndarray:
    """Per-point CRPS approximation; `quantiles` has shape (n_points, n_levels)."""
    quantiles = np.atleast_2d(np.asarray(quantiles, dtype=float))
    losses = np.column_stack([pinball_loss(actual, quantiles[:, j], tau) for j, tau in enumerate(levels)])
    return 2.0 * losses.mean(axis=1)


def interval_score(actual, lower, upper, alpha: float) -> np.ndarray:
    """Gneiting-Raftery interval score for a central (1 - alpha) interval; lower is better."""
    actual, lower, upper = (np.asarray(x, dtype=float) for x in (actual, lower, upper))
    return (
        (upper - lower)
        + (2.0 / alpha) * np.maximum(lower - actual, 0.0)
        + (2.0 / alpha) * np.maximum(actual - upper, 0.0)
    )


def seasonal_naive_quantiles(
    train: np.ndarray, horizon: int, levels: Sequence[float] = QUANTILE_LEVELS, season_length: int = 12,
) -> np.ndarray:
    """Seasonal Naive point plus empirical quantiles of its in-sample errors.

    Uses only the training series: errors are y[t] - y[t - season_length] within it.
    The same error distribution is applied at every lead, so the interval does not
    widen with horizon -- a deliberately simple baseline.
    """
    train = np.asarray(train, dtype=float)
    if train.size <= season_length:
        raise ValueError(f"need more than {season_length} observations")
    point = np.resize(train[-season_length:], horizon)
    residuals = train[season_length:] - train[:-season_length]
    offsets = np.quantile(residuals, levels)
    return point[:, None] + offsets[None, :]


def summarize_probabilistic(quantile_rows: pd.DataFrame, horizons: Sequence[int]) -> pd.DataFrame:
    """Per target x horizon x model metrics, averaged over origins like the point metrics."""
    q_cols = [quantile_column(t) for t in QUANTILE_LEVELS]
    lo, hi = quantile_column(INTERVAL_LOWER), quantile_column(INTERVAL_UPPER)
    alpha = 1.0 - (INTERVAL_UPPER - INTERVAL_LOWER)
    per_origin = []
    for horizon in horizons:
        window = quantile_rows.loc[quantile_rows["lead"] <= horizon]
        for (target, model, origin), group in window.groupby(["target", "model", "origin_number"]):
            actual = group["actual"].to_numpy(dtype=float)
            denom = np.abs(actual).sum()
            crps = quantile_crps(actual, group[q_cols].to_numpy())
            per_origin.append({
                "target": target,
                "horizon": horizon,
                "model": model,
                "origin_number": origin,
                "coverage_80": float(np.mean((actual >= group[lo]) & (actual <= group[hi])) * 100),
                "interval_width_pct": float((group[hi] - group[lo]).sum() / denom * 100),
                "interval_score_pct": float(interval_score(actual, group[lo], group[hi], alpha).sum() / denom * 100),
                "scaled_CRPS": float(crps.sum() / denom * 100),
            })
    per_origin = pd.DataFrame(per_origin)
    return (
        per_origin.groupby(["target", "horizon", "model"], as_index=False)
        [["coverage_80", "interval_width_pct", "interval_score_pct", "scaled_CRPS"]]
        .mean()
    )


def quantile_calibration(quantile_rows: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Share of actuals at or below each forecast quantile, pooled over targets and origins."""
    window = quantile_rows.loc[quantile_rows["lead"] <= horizon]
    rows = []
    for model, group in window.groupby("model"):
        for tau in QUANTILE_LEVELS:
            rows.append({
                "model": model,
                "horizon": horizon,
                "nominal_level": tau,
                "observed_share_below": float(np.mean(group["actual"] <= group[quantile_column(tau)])),
                "n": int(len(group)),
            })
    return pd.DataFrame(rows)
