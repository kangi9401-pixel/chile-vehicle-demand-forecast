import numpy as np
import pandas as pd
import pytest

from chile_forecast.probabilistic import (
    QUANTILE_LEVELS,
    interval_score,
    quantile_column,
    quantile_crps,
    seasonal_naive_quantiles,
    summarize_probabilistic,
)


def test_crps_of_a_point_forecast_is_the_absolute_error():
    actual = np.array([100.0, 80.0, 120.0])
    point = np.array([90.0, 95.0, 120.0])
    degenerate = np.repeat(point[:, None], len(QUANTILE_LEVELS), axis=1)
    np.testing.assert_allclose(quantile_crps(actual, degenerate), np.abs(actual - point))


def test_a_sharper_correctly_centred_distribution_has_lower_crps():
    actual = np.array([100.0])
    z = np.array([-1.645, -1.2816, -1.0364, -0.8416, -0.6745, -0.5244, -0.3853, -0.2533, -0.1257, 0.0,
                  0.1257, 0.2533, 0.3853, 0.5244, 0.6745, 0.8416, 1.0364, 1.2816, 1.645])
    sharp = 100 + 5 * z[None, :]
    wide = 100 + 50 * z[None, :]
    assert quantile_crps(actual, sharp)[0] < quantile_crps(actual, wide)[0]


def test_interval_score_is_width_inside_and_penalised_outside():
    assert interval_score([5.0], [0.0], [10.0], alpha=0.2)[0] == pytest.approx(10.0)
    # 2 units above the upper bound costs (2 / 0.2) * 2 = 20 on top of the width.
    assert interval_score([12.0], [0.0], [10.0], alpha=0.2)[0] == pytest.approx(30.0)


def test_seasonal_naive_quantiles_use_training_errors_only_and_are_monotone():
    rng = np.random.default_rng(0)
    train = 100 + 10 * np.sin(np.arange(48) * 2 * np.pi / 12) + rng.normal(0, 3, 48)
    q = seasonal_naive_quantiles(train, horizon=12)
    assert q.shape == (12, len(QUANTILE_LEVELS))
    assert (np.diff(q, axis=1) >= 0).all()
    residuals = train[12:] - train[:-12]
    np.testing.assert_allclose(q[0] - train[-12], np.quantile(residuals, QUANTILE_LEVELS))


def test_summary_coverage_counts_actuals_inside_the_80pct_interval():
    rows = []
    for lead, actual in enumerate([50.0, 150.0, 100.0], start=1):  # inside, above, inside
        rows.append({"target": "T", "model": "M", "origin_number": 1, "lead": lead, "actual": actual,
                     **{quantile_column(t): 60.0 + 80.0 * t for t in QUANTILE_LEVELS}})
    summary = summarize_probabilistic(pd.DataFrame(rows), horizons=(3,))
    # q10 = 68 and q90 = 132: 50 and 150 fall outside, 100 falls inside.
    assert summary.loc[0, "coverage_80"] == pytest.approx(100 / 3)
