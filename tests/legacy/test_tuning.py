import numpy as np
import pandas as pd

from chile_forecast.legacy.tuning import XGB_PARAM_DISTRIBUTIONS, tune_xgb_hyperparameters


def test_tune_xgb_hyperparameters_returns_valid_params():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(60, 3)), columns=["a", "b", "c"])
    y = (X["a"] * 2 + rng.normal(scale=0.1, size=60)).values

    best_params, results = tune_xgb_hyperparameters(X, y, n_splits=3, n_iter=4)

    assert set(best_params) <= set(XGB_PARAM_DISTRIBUTIONS)
    for key, value in best_params.items():
        assert value in XGB_PARAM_DISTRIBUTIONS[key]
    assert len(results) == 4
