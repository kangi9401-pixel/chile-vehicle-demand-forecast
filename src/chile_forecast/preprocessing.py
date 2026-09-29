"""Leakage-safe macro feature preprocessing.

The preprocessor is fitted independently inside every rolling-origin fold.  Raw
future macro values are never used by the forecast pipeline: callers transform the
training rows and then request a calendar-aware, last-observation-carried-forward
feature frame for the forecast horizon.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from chile_forecast.config import DATE_COL


RESOURCE_COLS = (
    "Crude_Oil_USD_per_Barrel",
    "Iron_Ore_USD_per_Ton",
    "Copper_USD_per_Ton",
)
RAW_NUMERIC_COLS = (
    "Middle",
    "High",
    "Population",
    "Interest Rate",
    *RESOURCE_COLS,
)


@dataclass
class LeakageSafePreprocessor:
    """Fit imputation/scaling statistics on training rows only."""

    medians_: Dict[str, float] = field(default_factory=dict, init=False)
    resource_means_: Dict[str, float] = field(default_factory=dict, init=False)
    resource_stds_: Dict[str, float] = field(default_factory=dict, init=False)
    feature_means_: Dict[str, float] = field(default_factory=dict, init=False)
    feature_stds_: Dict[str, float] = field(default_factory=dict, init=False)
    fitted_: bool = field(default=False, init=False)

    def fit(self, raw_train: pd.DataFrame) -> "LeakageSafePreprocessor":
        self._check_columns(raw_train, (DATE_COL, *RAW_NUMERIC_COLS))
        if raw_train.empty:
            raise ValueError("raw_train must contain at least one row")
        self.medians_ = {
            col: float(pd.to_numeric(raw_train[col], errors="coerce").median())
            for col in RAW_NUMERIC_COLS
        }
        self.resource_means_ = {
            col: float(pd.to_numeric(raw_train[col], errors="coerce").fillna(self.medians_[col]).mean())
            for col in RESOURCE_COLS
        }
        self.resource_stds_ = {}
        for col in RESOURCE_COLS:
            values = pd.to_numeric(raw_train[col], errors="coerce").fillna(self.medians_[col])
            std = float(values.std(ddof=0))
            self.resource_stds_[col] = std if np.isfinite(std) and std > 1e-12 else 1.0
        clean = raw_train.loc[:, RAW_NUMERIC_COLS].apply(pd.to_numeric, errors="coerce")
        for col in RAW_NUMERIC_COLS:
            clean[col] = clean[col].fillna(self.medians_[col])
        stable = pd.DataFrame({
            "middle_income": clean["Middle"],
            "interest_rate": clean["Interest Rate"],
            "middle_high_income_ratio": (
                (clean["Middle"] + clean["High"]) / clean["Population"].clip(lower=1.0)
            ),
        })
        self.feature_means_ = {col: float(stable[col].mean()) for col in stable}
        self.feature_stds_ = {}
        for col in stable:
            std = float(stable[col].std(ddof=0))
            self.feature_stds_[col] = std if np.isfinite(std) and std > 1e-12 else 1.0
        self.fitted_ = True
        return self

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("fit must be called before transform")
        self._check_columns(raw, (DATE_COL, *RAW_NUMERIC_COLS))
        work = raw.loc[:, [DATE_COL, *RAW_NUMERIC_COLS]].copy()
        for col in RAW_NUMERIC_COLS:
            work[col] = pd.to_numeric(work[col], errors="coerce").fillna(self.medians_[col])

        resource_z = pd.DataFrame(index=work.index)
        for col in RESOURCE_COLS:
            resource_z[col] = (work[col] - self.resource_means_[col]) / self.resource_stds_[col]

        population = work["Population"].clip(lower=1.0)
        dates = pd.DatetimeIndex(work[DATE_COL])
        middle_income = (work["Middle"] - self.feature_means_["middle_income"]) / self.feature_stds_["middle_income"]
        interest_rate = (work["Interest Rate"] - self.feature_means_["interest_rate"]) / self.feature_stds_["interest_rate"]
        income_ratio = (work["Middle"] + work["High"]) / population
        income_ratio = (income_ratio - self.feature_means_["middle_high_income_ratio"]) / self.feature_stds_["middle_high_income_ratio"]
        result = pd.DataFrame({
            DATE_COL: dates,
            "middle_income": middle_income.to_numpy(),
            "avg_resource_price": resource_z.mean(axis=1).to_numpy(),
            "interest_rate": interest_rate.to_numpy(),
            "middle_high_income_ratio": income_ratio.to_numpy(),
            "month_sin": np.sin(2 * np.pi * dates.month / 12),
            "month_cos": np.cos(2 * np.pi * dates.month / 12),
            "trend": np.arange(len(work), dtype=float),
        })
        # Ratios with avg_resource_price as denominator were intentionally removed:
        # a z-score crosses zero and makes those variables numerically unstable.
        numeric = result.columns.drop(DATE_COL)
        result[numeric] = result[numeric].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return result

    def fit_transform(self, raw_train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(raw_train).transform(raw_train)

    def make_future_features(self, raw_train: pd.DataFrame, future_dates: Iterable[pd.Timestamp]) -> pd.DataFrame:
        """Create features without consulting raw future covariate values."""
        if not self.fitted_:
            raise RuntimeError("fit must be called before make_future_features")
        future_dates = pd.DatetimeIndex(future_dates)
        last = raw_train.iloc[-1]
        frozen = pd.DataFrame({DATE_COL: future_dates})
        for col in RAW_NUMERIC_COLS:
            frozen[col] = last[col]
        transformed = self.transform(frozen)
        # Continue, rather than restart, the deterministic trend feature.
        transformed["trend"] = np.arange(len(raw_train), len(raw_train) + len(future_dates), dtype=float)
        return transformed

    @staticmethod
    def _check_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
        missing = sorted(set(columns) - set(frame.columns))
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
