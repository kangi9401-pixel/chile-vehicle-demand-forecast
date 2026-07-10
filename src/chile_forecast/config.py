"""Central configuration for the Chile vehicle demand forecasting pipeline."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "synthetic_chile_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
HOLDOUT_EXCEL_PATH = OUTPUT_DIR / "holdout_deepar_xgb_ensemble_compare.xlsx"

DATE_COL = "Date"

SEGMENT_COLS = [
    "Anac Market Size", "Total Market Size", "OEM_A", "OEM_B",
    "B-Sedan", "B_HB", "C_Sedan", "C_HB", "SUV-A", "SUV-B",
]

MACRO_COLS_FOR_DEEPAR = [
    "middle_income",
    "avg_resource_price",
    "interest_rate",
    "middle_high_income_ratio",
]

FREQ = "ME"
OBS_END = pd.Timestamp("2025-05-31")
PREDICTION_END_DATE = pd.Timestamp("2030-12-31")

FORECAST_STEP = 12
MIN_SERIES_LEN = 60
HOLDOUT_N = 3

BOLL_WINDOW = 20
BOLL_K = 2

# Hyperparameter search is run once on this representative segment (the aggregate
# market) rather than per segment -- with ~100 monthly observations per series,
# per-segment search would just overfit the search itself to each series' noise.
TUNING_SEGMENT = "Total Market Size"
VAL_N = 6  # months held out for DeepAR hyperparameter validation, just before HOLDOUT_N
