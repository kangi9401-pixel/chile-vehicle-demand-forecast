"""DeepAR/XGBoost ensemble averaging, holdout summary table, and fit-metric reporting."""

import logging
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from chile_forecast.config import FREQ, HOLDOUT_N
from chile_forecast.metrics import calc_fit_metrics

logger = logging.getLogger(__name__)


def compute_ensemble_weights(rolling_df: pd.DataFrame) -> Dict[str, float]:
    """Per-segment weight on DeepAR (XGBoost gets `1 - weight`), inverse-MAE weighted
    from the rolling-origin backtest -- i.e. whichever model was more accurate across
    the backtest origins for that segment gets more say, instead of an arbitrary 50/50
    split. Falls back to 0.5 for any segment missing from `rolling_df` or with a
    zero/undefined MAE."""
    weights: Dict[str, float] = {}
    if rolling_df.empty:
        return weights

    for seg, grp in rolling_df.groupby("Segment"):
        mae_deepar = grp["MAE_DeepAR"].mean()
        mae_xgb = grp["MAE_XGBoost"].mean()
        if not np.isfinite(mae_deepar) or not np.isfinite(mae_xgb) or (mae_deepar + mae_xgb) == 0:
            weights[seg] = 0.5
            continue
        inv_deepar, inv_xgb = 1 / max(mae_deepar, 1e-9), 1 / max(mae_xgb, 1e-9)
        weights[seg] = inv_deepar / (inv_deepar + inv_xgb)

    return weights


def build_holdout_row(
    seg: str,
    y_test: np.ndarray,
    preds_deepar: np.ndarray,
    preds_xgb: np.ndarray,
    holdout_start: pd.Timestamp,
    weight_deepar: float = 0.5,
) -> Dict:
    row: Dict = {"Segment": seg, "Weight_DeepAR": weight_deepar}

    min_len = min(len(y_test), len(preds_deepar), len(preds_xgb))
    y_test_eval = y_test[:min_len]
    preds_deepar_eval = preds_deepar[:min_len]
    preds_xgb_eval = preds_xgb[:min_len]
    preds_ensemble_eval = weight_deepar * preds_deepar_eval + (1 - weight_deepar) * preds_xgb_eval

    dates = pd.date_range(holdout_start, periods=HOLDOUT_N, freq=FREQ)

    for i, dt in enumerate(dates):
        month_str = f"{dt:%Y}M{dt.month:02d}"
        row[f"{month_str}_Actual"] = y_test[i] if i < len(y_test) else np.nan
        row[f"{month_str}_DeepAR"] = preds_deepar[i] if i < len(preds_deepar) else np.nan
        row[f"{month_str}_XGB"] = preds_xgb[i] if i < len(preds_xgb) else np.nan
        row[f"{month_str}_Ensemble"] = (
            weight_deepar * preds_deepar[i] + (1 - weight_deepar) * preds_xgb[i]
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
