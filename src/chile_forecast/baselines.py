"""Transparent forecasting baselines used in every backtest fold."""

from __future__ import annotations

from typing import Dict

import numpy as np


def last_value_naive(train: np.ndarray, horizon: int) -> np.ndarray:
    train = np.asarray(train, dtype=float)
    if train.size == 0:
        raise ValueError("train must not be empty")
    return np.repeat(train[-1], horizon)


def seasonal_naive(train: np.ndarray, horizon: int, season_length: int = 12) -> np.ndarray:
    """Repeat the last observed seasonal cycle (year-on-year for monthly data)."""
    train = np.asarray(train, dtype=float)
    if train.size < season_length:
        raise ValueError(f"seasonal naive requires at least {season_length} observations")
    cycle = train[-season_length:]
    return np.resize(cycle, horizon).astype(float)


def moving_average(train: np.ndarray, horizon: int, window: int = 6) -> np.ndarray:
    train = np.asarray(train, dtype=float)
    if train.size == 0:
        raise ValueError("train must not be empty")
    return np.repeat(float(np.mean(train[-min(window, train.size):])), horizon)


def baseline_predictions(train: np.ndarray, horizon: int) -> Dict[str, np.ndarray]:
    return {
        "LastValue": last_value_naive(train, horizon),
        "SeasonalNaive": seasonal_naive(train, horizon),
        "MovingAverage6": moving_average(train, horizon, window=6),
    }
