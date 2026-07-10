"""XGBoost holdout training/prediction on the macro feature set."""

from typing import Dict, Optional

import numpy as np
import pandas as pd
import xgboost as xgb

DEFAULT_XGB_PARAMS: Dict = dict(
    objective="reg:squarederror",
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
)


def run_xgb_holdout(
    X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame,
    xgb_params: Optional[Dict] = None,
) -> np.ndarray:
    params = {**DEFAULT_XGB_PARAMS, **(xgb_params or {})}
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def fit_xgb(X_train: pd.DataFrame, y_train: np.ndarray, xgb_params: Optional[Dict] = None) -> xgb.XGBRegressor:
    """Like `run_xgb_holdout` but returns the fitted model itself (for SHAP analysis, etc.)."""
    params = {**DEFAULT_XGB_PARAMS, **(xgb_params or {})}
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model
