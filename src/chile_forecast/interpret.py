"""SHAP-based feature importance for the XGBoost leg of the ensemble.

DeepAR doesn't have an established SHAP-style attribution method (it's an
autoregressive RNN over a learned latent state, not a feature-additive model), so
this covers the interpretable half of the ensemble: which macro covariates XGBoost
actually leans on, and in which direction.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap
import xgboost as xgb

logger = logging.getLogger(__name__)


def shap_summary_plot(
    model: xgb.XGBRegressor, X: pd.DataFrame, seg: str, output_dir: Path,
) -> Optional[Path]:
    """Fit a TreeExplainer on `model` and save a beeswarm summary plot of SHAP
    values over `X` (typically the training features for `seg`)."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    fig = plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, X, show=False, plot_size=None)
    plt.title(f"{seg} - XGBoost SHAP Feature Importance")
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{seg.replace(' ', '_').replace('/', '-')}_shap_summary.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("[%s] SHAP summary plot saved to %s", seg, out_path)
    return out_path


def mean_abs_shap_importance(model: xgb.XGBRegressor, X: pd.DataFrame) -> pd.Series:
    """Mean |SHAP value| per feature, sorted descending -- a quick tabular summary
    to pair with the plot (and to embed directly in the technical report)."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    importance = pd.Series(
        abs(shap_values.values).mean(axis=0), index=X.columns, name="mean_abs_shap"
    ).sort_values(ascending=False)
    return importance
