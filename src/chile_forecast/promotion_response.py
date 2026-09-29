"""Clearly labelled synthetic promotion-response inputs.

Nothing in this module represents a real manufacturer, a dealer, or an observed causal effect.
The numbers are pedagogical assumptions in million Chilean pesos (MCLP).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


MODEL_TARGET_MAP: Dict[str, str] = {
    "City Hatchback": "B_HB",
    "Compact Sedan": "B-Sedan",
    "Compact SUV": "SUV-A",
    "Family SUV": "SUV-B",
}


def _latest_forecast_by_target(predictions: pd.DataFrame, summary: pd.DataFrame) -> Dict[str, float]:
    """Select the latest 3-month forecast using strictly earlier origins."""
    latest_origin = predictions["origin_number"].max()
    history = predictions.loc[
        (predictions["horizon"] == 3) & (predictions["origin_number"] < latest_origin)
    ].copy()
    if history.empty:
        # Small unit-test/user-supplied inputs may contain only one origin.
        best = (
            summary.loc[summary["horizon"] == 3]
            .sort_values(["target", "WAPE_mean", "model"])
            .groupby("target", as_index=False)
            .first()[["target", "model"]]
        )
    else:
        history["abs_error"] = (history["actual"] - history["predicted"]).abs()
        score = history.groupby(["target", "model"], as_index=False).agg(
            abs_error=("abs_error", "sum"), actual_abs=("actual", lambda x: x.abs().sum())
        )
        score["historical_WAPE"] = score["abs_error"] / score["actual_abs"].replace(0, np.nan)
        best = (
            score.sort_values(["target", "historical_WAPE", "model"])
            .groupby("target", as_index=False)
            .first()[["target", "model"]]
        )
    rows = predictions.loc[
        (predictions["origin_number"] == latest_origin) & (predictions["horizon"] == 3)
    ].merge(best, on=["target", "model"], how="inner")
    return rows.groupby("target")["predicted"].sum().to_dict()


def build_synthetic_promotion_inputs(
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build five synthetic model rows and their diminishing-return tranches."""
    forecast = _latest_forecast_by_target(predictions, summary)
    defaults = {"B_HB": 2200.0, "B-Sedan": 2400.0, "SUV-A": 2000.0, "SUV-B": 1800.0}
    base = {key: max(float(forecast.get(key, value)), 100.0) for key, value in defaults.items()}

    specs = [
        # model, target, factory, inventory multiplier, supply, capacity, lifecycle,
        # price, pre-tariff cost, tariff, variable cost, regulation, strategy, min, max, prior
        ("City Hatchback", "B_HB", "Plant North", 0.62, 1600, 2600, "mature", 13.8, 7.0, 0.06, 1.7, True, 1.00, 0, 105, 2150),
        ("Compact Sedan", "B-Sedan", "Plant North", 0.78, 1450, 2600, "phase-out", 16.2, 8.3, 0.06, 2.0, True, 1.08, 30, 115, 2450),
        ("Compact SUV", "SUV-A", "Plant South", 0.55, 1800, 2980, "growth", 20.5, 10.7, 0.06, 2.4, True, 1.18, 45, 135, 1880),
        ("Family SUV", "SUV-B", "Plant South", 0.50, 1450, 2980, "mature", 24.0, 13.0, 0.06, 2.8, True, 1.05, 15, 125, 1620),
        ("Legacy Diesel SUV", None, "Plant South", 0.0, 0, 2980, "phase-out", 22.0, 12.5, 0.08, 2.7, False, 0.85, 0, 0, 420),
    ]
    rows = []
    for model, target, factory, inv_mult, supply, factory_cap, lifecycle, price, cost, tariff, var_cost, allowed, strategy, minimum, maximum, prior in specs:
        raw_demand = 0.0 if target is None else base[target]
        inventory = int(round(raw_demand * inv_mult))
        available = inventory + supply
        # Base planning reserves 10% sellable capacity; the upside scenario may
        # consume that reserve down to 4%, making plant constraints informative.
        base_demand = min(raw_demand, available * 0.90)
        tariff_adjusted_cost = cost * (1 + tariff)
        margin = price - tariff_adjusted_cost - var_cost
        rows.append({
            "model": model,
            "forecast_target": target or "not_applicable",
            "factory": factory,
            "base_demand": round(base_demand, 2),
            "forecast_lower": round(base_demand * 0.85, 2),
            "forecast_upper": round(base_demand * 1.15, 2),
            "inventory": inventory,
            "supply_available": supply,
            "factory_capacity": factory_cap,
            "lifecycle": lifecycle,
            "unit_revenue_mclp": price,
            "pre_tariff_cost_mclp": cost,
            "tariff_rate": tariff,
            "tariff_adjusted_cost_mclp": round(tariff_adjusted_cost, 3),
            "unit_margin_mclp": round(margin, 3),
            "regulatory_allowed": allowed,
            "strategic_weight": strategy,
            "min_budget_mclp": minimum,
            "max_budget_mclp": maximum,
            "prior_year_sales": prior,
        })
    models = pd.DataFrame(rows)

    # Factory caps are synthetic but internally coherent across different forecast
    # magnitudes: at the upside cap (96% of available units), each plant retains a
    # small shared residual that makes the constraint binding without making the
    # declared strategic minimum budgets infeasible.
    for factory, idx in models.groupby("factory").groups.items():
        group = models.loc[idx]
        max_base_production = np.maximum(
            0.96 * (group["inventory"] + group["supply_available"]) - group["inventory"],
            0.0,
        ).sum()
        models.loc[idx, "factory_capacity"] = float(np.ceil(max_base_production + 25.0))

    # Units gained per additional MCLP.  Slopes strictly decline by tranche.
    response_specs = {
        "City Hatchback": (0.31, 0.20, 0.10),
        "Compact Sedan": (0.29, 0.18, 0.08),
        "Compact SUV": (0.34, 0.22, 0.11),
        "Family SUV": (0.27, 0.17, 0.08),
        "Legacy Diesel SUV": (0.00, 0.00, 0.00),
    }
    tranches = []
    for model, slopes in response_specs.items():
        for number, (width, slope) in enumerate(zip((35.0, 45.0, 55.0), slopes), start=1):
            tranches.append({
                "model": model,
                "tranche": number,
                "width_mclp": width,
                "incremental_units_per_mclp": slope,
            })
    return models, pd.DataFrame(tranches)


def incremental_units(budget_mclp: float, response_rows: pd.DataFrame) -> float:
    remaining = max(float(budget_mclp), 0.0)
    units = 0.0
    for row in response_rows.sort_values("tranche").itertuples():
        used = min(remaining, row.width_mclp)
        units += used * row.incremental_units_per_mclp
        remaining -= used
        if remaining <= 1e-12:
            break
    return float(units)
