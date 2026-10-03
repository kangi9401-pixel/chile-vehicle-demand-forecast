"""Diebold-Mariano tests on per-origin forecast losses.

The loss for each origin is that origin's WAPE over the horizon.  With origins three
months apart, an h-month window overlaps the next ceil(h/3) - 1 windows, so the loss
differentials are autocorrelated up to that lag.  The variance uses a Newey-West
(Bartlett) estimator with that lag, which stays positive in small samples, and the
statistic gets the Harvey-Leybourne-Newbold small-sample correction with a Student-t
reference distribution.  With only eight origins per segment the tests have low power:
a large p-value means "not distinguishable here", not "equally accurate".
"""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from chile_forecast.config import BACKTEST_STEP_MONTHS

COMPARISONS: Tuple[Tuple[str, str], ...] = (
    ("LeakageSafeEnsemble", "SeasonalNaive"),
    ("LeakageSafeEnsemble", "XGBoost"),
    ("LeakageSafeEnsemble", "DeepAR"),
    ("XGBoost", "SeasonalNaive"),
    ("DeepAR", "SeasonalNaive"),
    ("DeepAR", "XGBoost"),
)
ALL_TARGETS = "All segments"


def overlap_lag(horizon: int, step_months: int = BACKTEST_STEP_MONTHS) -> int:
    """Number of following origins whose evaluation windows overlap this one."""
    return max(math.ceil(horizon / step_months) - 1, 0)


def diebold_mariano(differential: Sequence[float], lag: int) -> Dict[str, float]:
    """Two-sided DM test of mean(differential) == 0 with HLN correction."""
    d = np.asarray(differential, dtype=float)
    n = d.size
    mean = float(d.mean()) if n else np.nan
    if n < 3:
        return {"n": n, "mean_diff": mean, "dm_stat": np.nan, "p_value": np.nan}
    centred = d - mean
    variance = float(np.dot(centred, centred) / n)
    for k in range(1, min(lag, n - 1) + 1):
        weight = 1.0 - k / (lag + 1.0)
        variance += 2.0 * weight * float(np.dot(centred[k:], centred[:-k]) / n)
    if variance <= 0:
        return {"n": n, "mean_diff": mean, "dm_stat": np.nan, "p_value": np.nan}
    h = lag + 1
    correction = math.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 0.0))
    statistic = correction * mean / math.sqrt(variance / n)
    p_value = float(2.0 * stats.t.sf(abs(statistic), df=n - 1))
    return {"n": n, "mean_diff": mean, "dm_stat": float(statistic), "p_value": p_value}


def _holm(p_values: pd.Series) -> pd.Series:
    """Holm step-down adjustment; NaNs are left untouched and not counted."""
    valid = p_values.dropna().sort_values()
    m = len(valid)
    adjusted, running = {}, 0.0
    for rank, (index, p) in enumerate(valid.items()):
        running = max(running, min((m - rank) * p, 1.0))
        adjusted[index] = running
    return pd.Series(adjusted, dtype=float).reindex(p_values.index)


def forecast_significance(
    metrics_by_origin: pd.DataFrame,
    comparisons: Sequence[Tuple[str, str]] = COMPARISONS,
    loss: str = "WAPE",
) -> pd.DataFrame:
    """DM tests per segment and for the segment average, per horizon.

    ``mean_diff`` is loss(model_a) - loss(model_b) in WAPE points, so a negative value
    favours ``model_a``.  ``p_holm`` adjusts for the comparisons within each
    (scope, horizon) family.
    """
    available = set(metrics_by_origin["model"])
    comparisons = [(a, b) for a, b in comparisons if a in available and b in available]
    wide = metrics_by_origin.pivot_table(
        index=["target", "horizon", "origin_number"], columns="model", values=loss
    )
    pooled = wide.groupby(level=["horizon", "origin_number"]).mean()
    pooled = pd.concat({ALL_TARGETS: pooled}, names=["target"])
    wide = pd.concat([wide, pooled])

    rows = []
    for (target, horizon), group in wide.groupby(level=["target", "horizon"]):
        group = group.sort_index(level="origin_number")
        lag = overlap_lag(int(horizon))
        for model_a, model_b in comparisons:
            result = diebold_mariano((group[model_a] - group[model_b]).to_numpy(), lag)
            rows.append({
                "scope": target, "horizon": int(horizon), "model_a": model_a, "model_b": model_b,
                "hac_lag": lag, **result,
            })
    result = pd.DataFrame(rows)
    result["p_holm"] = result.groupby(["scope", "horizon"], group_keys=False)["p_value"].apply(_holm)
    order = {name: i for i, name in enumerate(sorted(set(result["scope"]) - {ALL_TARGETS}) + [ALL_TARGETS])}
    return result.sort_values(["scope", "horizon"], key=lambda col: col.map(order) if col.name == "scope" else col).reset_index(drop=True)
