import numpy as np
import pandas as pd
import pytest

from chile_forecast.evaluation import _inverse_mae_weight, _visible_history


def _record(start, deepar_errors, xgb_errors):
    dates = pd.date_range(start, periods=len(deepar_errors), freq="ME")
    return dates, np.asarray(deepar_errors, dtype=float), np.asarray(xgb_errors, dtype=float)


def test_errors_after_cutoff_cannot_change_visible_history():
    cutoff = pd.Timestamp("2024-02-29")
    clean = [_record("2024-01-31", [10, 20, 30, 40], [5, 5, 5, 5])]
    poisoned = [_record("2024-01-31", [10, 20, 1e12, 1e12], [5, 5, -1e12, 1e12])]

    assert _visible_history(clean, cutoff) == _visible_history(poisoned, cutoff)
    assert _visible_history(clean, cutoff) == [(15.0, 5.0)]


def test_only_dates_up_to_cutoff_count_and_unseen_origins_are_dropped():
    cutoff = pd.Timestamp("2024-02-29")
    records = [
        _record("2023-11-30", [1, 2, 3, 4, 5], [2, 2, 2, 2, 2]),  # visible through 2024-02
        _record("2024-03-31", [100, 100], [100, 100]),            # nothing visible yet
    ]

    assert _visible_history(records, cutoff) == [(pytest.approx(2.5), pytest.approx(2.0))]
    assert _visible_history(records[1:], cutoff) == []


def test_empty_history_gives_equal_weight():
    assert _inverse_mae_weight([]) == 0.5
    assert _inverse_mae_weight(_visible_history([], pd.Timestamp("2024-02-29"))) == 0.5
