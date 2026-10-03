import numpy as np
import pandas as pd
import pytest
from scipy import stats

from chile_forecast.significance import ALL_TARGETS, diebold_mariano, forecast_significance, overlap_lag


def test_overlap_lag_matches_three_month_origin_step():
    assert [overlap_lag(h) for h in (3, 6, 12)] == [0, 1, 3]


def test_dm_without_overlap_equals_one_sample_t_test():
    d = np.array([-1.2, 0.4, -2.0, -0.7, 0.1, -1.5, -0.3, -0.9])
    result = diebold_mariano(d, lag=0)
    t = stats.ttest_1samp(d, 0.0)
    assert result["dm_stat"] == pytest.approx(t.statistic)
    assert result["p_value"] == pytest.approx(t.pvalue)


def test_dm_is_antisymmetric_and_undefined_for_zero_variance():
    d = np.array([-1.0, 0.5, -2.0, -0.5, 0.2, -1.1, -0.4, -0.8])
    assert diebold_mariano(d, 3)["dm_stat"] == pytest.approx(-diebold_mariano(-d, 3)["dm_stat"])
    assert np.isnan(diebold_mariano(np.ones(8), 1)["p_value"])


def test_forecast_significance_pools_segments_and_signs_the_difference():
    rows = []
    for target, offset in (("A", 0.0), ("B", 1.0)):
        for origin in range(1, 9):
            for model, wape in (("XGBoost", 10.0 + offset + 0.1 * origin), ("SeasonalNaive", 12.0 + offset + 0.3 * (origin % 3))):
                rows.append({"target": target, "horizon": 3, "origin_number": origin, "model": model, "WAPE": wape})
    result = forecast_significance(pd.DataFrame(rows))
    assert set(result["scope"]) == {"A", "B", ALL_TARGETS}
    assert len(result) == 3  # only the XGBoost vs SeasonalNaive pair is available
    pooled = result.loc[result["scope"] == ALL_TARGETS].iloc[0]
    assert pooled["mean_diff"] < 0 and pooled["p_value"] < 0.05
