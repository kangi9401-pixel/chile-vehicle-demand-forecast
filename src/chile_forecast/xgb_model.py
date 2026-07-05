"""XGBoost holdout training/prediction on the macro feature set."""

import numpy as np
import pandas as pd
import xgboost as xgb


def run_xgb_holdout(
    X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame,
) -> np.ndarray:
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model.predict(X_test)
