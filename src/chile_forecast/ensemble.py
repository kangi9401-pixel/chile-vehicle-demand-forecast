"""DeepAR/XGBoost ensemble averaging, holdout summary table, and fit-metric reporting."""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from chile_forecast.config import HOLDOUT_N
from chile_forecast.metrics import calc_fit_metrics

logger = logging.getLogger(__name__)


def build_holdout_row(
    seg: str,
    y_test: np.ndarray,
    preds_deepar: np.ndarray,
    preds_xgb: np.ndarray,
    holdout_start: pd.Timestamp,
) -> Dict:
    row: Dict = {"Segment": seg}

    min_len = min(len(y_test), len(preds_deepar), len(preds_xgb))
    y_test_eval = y_test[:min_len]
    preds_deepar_eval = preds_deepar[:min_len]
    preds_xgb_eval = preds_xgb[:min_len]
    preds_ensemble_eval = (preds_deepar_eval + preds_xgb_eval) / 2

    dates = pd.date_range(holdout_start, periods=HOLDOUT_N, freq="ME")

    for i, dt in enumerate(dates):
        month_str = f"{dt:%Y}M{dt.month:02d}"
        row[f"{month_str}_Actual"] = y_test[i] if i < len(y_test) else np.nan
        row[f"{month_str}_DeepAR"] = preds_deepar[i] if i < len(preds_deepar) else np.nan
        row[f"{month_str}_XGB"] = preds_xgb[i] if i < len(preds_xgb) else np.nan
        row[f"{month_str}_Ensemble"] = (
            (preds_deepar[i] + preds_xgb[i]) / 2
            if i < len(preds_deepar) and i < len(preds_xgb)
            else np.nan
        )

    valid_mask = (
        ~np.isnan(y_test_eval)
        & ~np.isnan(preds_deepar_eval)
        & ~np.isnan(preds_xgb_eval)
        & ~np.isnan(preds_ensemble_eval)
    )

    if np.sum(valid_mask) > 0:
        row["MAE_DeepAR"] = mean_absolute_error(y_test_eval[valid_mask], preds_deepar_eval[valid_mask])
        row["MAE_XGBoost"] = mean_absolute_error(y_test_eval[valid_mask], preds_xgb_eval[valid_mask])
        row["MAE_Ensemble"] = mean_absolute_error(y_test_eval[valid_mask], preds_ensemble_eval[valid_mask])
    else:
        row["MAE_DeepAR"] = row["MAE_XGBoost"] = row["MAE_Ensemble"] = np.nan

    return row


def print_anac_fit(holdout_df: pd.DataFrame, target_segment: str = "Anac Market Size") -> None:
    if "Segment" not in holdout_df.columns:
        logger.info("Holdout: no 'Segment' column present")
        return

    anac = holdout_df[holdout_df["Segment"] == target_segment]
    if anac.empty:
        logger.info("Holdout: no rows for segment '%s'", target_segment)
        return

    # Anchor on the "<year>M<month>_<Suffix>" per-period columns so this doesn't
    # also match the "MAE_DeepAR"/"MAE_XGBoost"/"MAE_Ensemble" summary columns,
    # which contain the same suffix as a substring.
    month_col = r"^\d{4}M\d{2}_"
    y_true = anac.filter(regex=month_col + "Actual$").values.flatten()
    y_deepar = anac.filter(regex=month_col + "DeepAR$").values.flatten()
    y_xgb = anac.filter(regex=month_col + "XGB$").values.flatten()
    y_ensemble = anac.filter(regex=month_col + "Ensemble$").values.flatten()

    logger.info("=== %s Fit Metrics ===", target_segment)
    logger.info("DeepAR:   %s", calc_fit_metrics(y_true, y_deepar))
    logger.info("XGBoost:  %s", calc_fit_metrics(y_true, y_xgb))
    logger.info("Ensemble: %s", calc_fit_metrics(y_true, y_ensemble))
