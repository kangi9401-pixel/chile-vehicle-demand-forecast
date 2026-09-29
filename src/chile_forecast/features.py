"""Series utilities and macro feature engineering."""

from typing import Tuple

import numpy as np
import pandas as pd
from gluonts.time_feature import time_features_from_frequency_str

from chile_forecast.config import DATE_COL, FREQ


def get_longest_non_nan_slice(series: pd.Series) -> Tuple[int, int, int]:
    """Return (start, end, length) of the longest contiguous non-NaN run in `series`."""
    max_len = 0
    best_start = best_end = 0
    cur_start = None

    for i, v in enumerate(series):
        if not pd.isna(v):
            if cur_start is None:
                cur_start = i
        else:
            if cur_start is not None and i - cur_start > max_len:
                max_len = i - cur_start
                best_start = cur_start
                best_end = i
            cur_start = None

    if cur_start is not None and len(series) - cur_start > max_len:
        max_len = len(series) - cur_start
        best_start = cur_start
        best_end = len(series)

    return best_start, best_end, best_end - best_start


def create_time_features(dates: pd.Series, freq: str = FREQ) -> np.ndarray:
    feats = time_features_from_frequency_str(freq)
    return np.stack([f(pd.DatetimeIndex(dates)) for f in feats], axis=0)


def build_feature_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Derive the macro feature set (income ratios, resource-price ratios, etc.)
    used as DeepAR/XGBoost covariates by the legacy (pre-audit) workflow, `run_legacy`."""
    raw_df = raw_df.copy()

    # Crude oil (~USD/barrel, tens), iron ore (~USD/ton, tens-hundreds) and copper
    # (~USD/ton, thousands) live on very different scales, so an unweighted mean is
    # dominated almost entirely by copper. Z-score each series first so the
    # composite actually reflects all three commodities rather than being a
    # thin proxy for the copper price alone.
    resource_cols = ["Crude_Oil_USD_per_Barrel", "Iron_Ore_USD_per_Ton", "Copper_USD_per_Ton"]
    resource_z = (raw_df[resource_cols] - raw_df[resource_cols].mean()) / raw_df[resource_cols].std()
    raw_df["avg_resource_price"] = resource_z.mean(axis=1)

    raw_df["middle_high_income_ratio"] = (
        raw_df["Middle"] + raw_df["High"]
    ) / raw_df["Population"]

    df = pd.DataFrame({
        DATE_COL: raw_df[DATE_COL],
        "middle_income": raw_df["Middle"],
        "avg_resource_price": raw_df["avg_resource_price"],
        "interest_rate": raw_df["Interest Rate"],
        "middle_high_income_ratio": raw_df["middle_high_income_ratio"],
    })

    df["middle_income/avg_resource_price_ratio"] = (
        df["middle_income"] / df["avg_resource_price"]
    )
    df["middle_high_income_ratio/avg_resource_price_ratio"] = (
        df["middle_high_income_ratio"] / df["avg_resource_price"]
    )
    df["interest_rate/middle_high_income_ratio_ratio"] = (
        df["interest_rate"] / df["middle_high_income_ratio"]
    )
    df["interest_rate/avg_resource_price_ratio"] = (
        df["interest_rate"] / df["avg_resource_price"]
    )
    df["interest_rate/middle_income_ratio"] = (
        df["interest_rate"] / df["middle_income"]
    )
    df["interest_rate+middle_high_income_ratio_mean"] = (
        df["interest_rate"] + df["middle_high_income_ratio"]
    ) / 2

    feature_cols = df.columns.drop(DATE_COL)
    df[feature_cols] = df[feature_cols].ffill().fillna(0)
    df.replace([np.inf, -np.inf], 0, inplace=True)

    return df
