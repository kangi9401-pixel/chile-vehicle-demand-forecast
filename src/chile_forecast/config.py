"""Central configuration for the Chile vehicle demand forecasting pipeline."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "synthetic_chile_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DECISION_OUTPUT_DIR = OUTPUT_DIR / "decision_system"

DATE_COL = "Date"
FREQ = "ME"

# Leakage-safe macro covariates produced by `LeakageSafePreprocessor` and passed to
# DeepAR as dynamic features.
MACRO_COLS_FOR_DEEPAR = [
    "middle_income",
    "avg_resource_price",
    "interest_rate",
    "middle_high_income_ratio",
]

FORECAST_STEP = 12  # DeepAR context length is 2 * FORECAST_STEP months (capped by history)

# Reproducible, common evaluation design used by the portfolio pipeline.
RANDOM_SEED = 42
BACKTEST_HORIZONS = (3, 6, 12)
BACKTEST_ORIGINS = 8
BACKTEST_STEP_MONTHS = 3
BACKTEST_TARGETS = ("B-Sedan", "B_HB", "SUV-A", "SUV-B")

# DeepAR is deliberately small: the dataset contains only about one hundred monthly
# observations per series.  More capacity would add runtime and overfitting risk, not
# credible evidence.  Override with `--deepar-epochs`; use `--skip-deepar` for a fast
# smoke test.
DEEPAR_BACKTEST_EPOCHS = 2

# --- Legacy workflow only (`chile_forecast.legacy`) -------------------------------
HOLDOUT_EXCEL_PATH = OUTPUT_DIR / "holdout_deepar_xgb_ensemble_compare.xlsx"

SEGMENT_COLS = [
    "Anac Market Size", "Total Market Size", "OEM_A", "OEM_B",
    "B-Sedan", "B_HB", "C_Sedan", "C_HB", "SUV-A", "SUV-B",
]

OBS_END = pd.Timestamp("2025-05-31")
PREDICTION_END_DATE = pd.Timestamp("2030-12-31")

MIN_SERIES_LEN = 60
HOLDOUT_N = 3

# Hyperparameter search is run once on this representative segment (the aggregate
# market) rather than per segment -- with ~100 monthly observations per series,
# per-segment search would just overfit the search itself to each series' noise.
TUNING_SEGMENT = "Total Market Size"
VAL_N = 6  # months held out for DeepAR hyperparameter validation, just before HOLDOUT_N
