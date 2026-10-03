"""Promotion-budget allocation with a transparent piecewise-linear LP."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from chile_forecast.promotion_response import incremental_units


TOTAL_BUDGET_MCLP = 300.0
SCENARIOS = {"downside": 0.85, "base": 1.00, "upside": 1.15}


def _effective_models(models: pd.DataFrame, demand_factor: float) -> pd.DataFrame:
    work = models.copy()
    work["scenario_base_demand"] = np.minimum(
        work["base_demand"] * demand_factor,
        0.96 * (work["inventory"] + work["supply_available"]),
    )
    work.loc[~work["regulatory_allowed"], "scenario_base_demand"] = 0.0
    return work


def optimize_budget(
    models: pd.DataFrame,
    response: pd.DataFrame,
    budget_mclp: float = TOTAL_BUDGET_MCLP,
    demand_factor: float = 1.0,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Maximize strategy-weighted incremental profit net of promotion spend."""
    work = _effective_models(models, demand_factor).reset_index(drop=True)
    variables = response.merge(
        work[["model", "factory", "unit_margin_mclp", "strategic_weight", "lifecycle", "regulatory_allowed"]],
        on="model",
        how="left",
    ).reset_index(drop=True)
    aged_bonus = variables["lifecycle"].map({"phase-out": 0.45, "mature": 0.10, "growth": 0.0}).fillna(0.0)
    marginal_value = (
        variables["incremental_units_per_mclp"]
        * (variables["unit_margin_mclp"] + aged_bonus)
        * variables["strategic_weight"]
        - 1.0
    )
    c = -marginal_value.to_numpy(dtype=float)
    bounds = [
        (0.0, float(row.width_mclp) if bool(row.regulatory_allowed) else 0.0)
        for row in variables.itertuples()
    ]
    A_ub, b_ub = [], []
    A_ub.append(np.ones(len(variables)))
    b_ub.append(float(budget_mclp))

    for model_row in work.itertuples():
        mask = (variables["model"] == model_row.model).to_numpy(dtype=float)
        A_ub.append(mask)
        b_ub.append(float(model_row.max_budget_mclp))
        if model_row.min_budget_mclp > 0:
            A_ub.append(-mask)
            b_ub.append(-float(model_row.min_budget_mclp))
        capacity = max(
            float(model_row.inventory + model_row.supply_available - model_row.scenario_base_demand),
            0.0,
        )
        response_coeff = mask * variables["incremental_units_per_mclp"].to_numpy(dtype=float)
        A_ub.append(response_coeff)
        b_ub.append(capacity)

    for factory, group in work.groupby("factory"):
        var_mask = (variables["factory"] == factory).to_numpy(dtype=float)
        coeff = var_mask * variables["incremental_units_per_mclp"].to_numpy(dtype=float)
        base_production = np.maximum(group["scenario_base_demand"] - group["inventory"], 0).sum()
        factory_capacity = float(group["factory_capacity"].iloc[0])
        A_ub.append(coeff)
        b_ub.append(max(factory_capacity - float(base_production), 0.0))

    result = linprog(c, A_ub=np.asarray(A_ub), b_ub=np.asarray(b_ub), bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"promotion optimization failed: {result.message}")
    variables["allocated_mclp"] = result.x
    allocation = variables.groupby("model", as_index=False)["allocated_mclp"].sum()
    allocation = work[["model"]].merge(allocation, on="model", how="left").fillna({"allocated_mclp": 0.0})
    return allocation, {
        "solver_success": bool(result.success),
        "weighted_incremental_objective_mclp": float(-result.fun),
        "budget_limit_mclp": float(budget_mclp),
    }


def _proportional_allocation(models: pd.DataFrame, weights: np.ndarray, budget: float) -> pd.DataFrame:
    allowed = models["regulatory_allowed"].to_numpy(dtype=bool)
    lower = np.where(allowed, models["min_budget_mclp"], 0.0).astype(float)
    upper = np.where(allowed, models["max_budget_mclp"], 0.0).astype(float)
    allocation = lower.copy()
    remaining = min(float(budget), float(upper.sum())) - float(allocation.sum())
    weights = np.where(allowed, np.maximum(np.asarray(weights, dtype=float), 0.0), 0.0)
    while remaining > 1e-9:
        room = upper - allocation
        active = room > 1e-9
        if not np.any(active):
            break
        active_weights = np.where(active, weights, 0.0)
        if active_weights.sum() <= 0:
            active_weights = active.astype(float)
        proposed = remaining * active_weights / active_weights.sum()
        added = np.minimum(proposed, room)
        allocation += added
        new_remaining = remaining - float(added.sum())
        if new_remaining >= remaining - 1e-10:
            break
        remaining = new_remaining
    return pd.DataFrame({"model": models["model"], "allocated_mclp": allocation})


def heuristic_allocations(models: pd.DataFrame, budget_mclp: float = TOTAL_BUDGET_MCLP) -> Dict[str, pd.DataFrame]:
    return {
        "Equal": _proportional_allocation(models, np.ones(len(models)), budget_mclp),
        "ForecastShare": _proportional_allocation(models, models["base_demand"].to_numpy(), budget_mclp),
        "PriorYearShare": _proportional_allocation(models, models["prior_year_sales"].to_numpy(), budget_mclp),
    }


def evaluate_allocation(
    strategy: str,
    allocation: pd.DataFrame,
    models: pd.DataFrame,
    response: pd.DataFrame,
    demand_factor: float = 1.0,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    work = _effective_models(models, demand_factor).merge(allocation, on="model", how="left")
    work["allocated_mclp"] = work["allocated_mclp"].fillna(0.0)
    raw_incremental = []
    for row in work.itertuples():
        response_rows = response.loc[response["model"] == row.model]
        units = incremental_units(row.allocated_mclp, response_rows)
        capacity = max(row.inventory + row.supply_available - row.scenario_base_demand, 0.0)
        raw_incremental.append(min(units, capacity) if row.regulatory_allowed else 0.0)
    work["expected_incremental_sales"] = raw_incremental

    # Enforce shared plant capacity for heuristic as well as optimized strategies.
    for factory, idx in work.groupby("factory").groups.items():
        rows = work.loc[idx]
        base_production = np.maximum(rows["scenario_base_demand"] - rows["inventory"], 0).sum()
        residual = max(float(rows["factory_capacity"].iloc[0]) - float(base_production), 0.0)
        total_inc = float(rows["expected_incremental_sales"].sum())
        if total_inc > residual and total_inc > 0:
            work.loc[idx, "expected_incremental_sales"] *= residual / total_inc

    work["expected_sales"] = work["scenario_base_demand"] + work["expected_incremental_sales"]
    work["expected_revenue_mclp"] = work["expected_sales"] * work["unit_revenue_mclp"]
    work["expected_contribution_mclp"] = (
        work["expected_sales"] * work["unit_margin_mclp"] - work["allocated_mclp"]
    )
    work["expected_incremental_profit_mclp"] = (
        work["expected_incremental_sales"] * work["unit_margin_mclp"] - work["allocated_mclp"]
    )
    work["inventory_depletion_units"] = np.minimum(work["expected_sales"], work["inventory"])
    work.insert(0, "strategy", strategy)

    budget_ok = float(work["allocated_mclp"].sum()) <= TOTAL_BUDGET_MCLP + 1e-6
    bounds_ok = bool(((work["allocated_mclp"] >= work["min_budget_mclp"] - 1e-6) & (work["allocated_mclp"] <= work["max_budget_mclp"] + 1e-6)).all())
    supply_ok = bool((work["expected_sales"] <= work["inventory"] + work["supply_available"] + 1e-6).all())
    regulation_ok = bool((work.loc[~work["regulatory_allowed"], "allocated_mclp"].abs() <= 1e-6).all())
    factory_ok = True
    for _, group in work.groupby("factory"):
        production = np.maximum(group["expected_sales"] - group["inventory"], 0).sum()
        factory_ok &= production <= float(group["factory_capacity"].iloc[0]) + 1e-6
    checks = {
        "strategy": strategy,
        "budget_ok": budget_ok,
        "model_budget_bounds_ok": bounds_ok,
        "supply_inventory_ok": supply_ok,
        "factory_capacity_ok": bool(factory_ok),
        "regulation_ok": regulation_ok,
        "all_constraints_ok": bool(budget_ok and bounds_ok and supply_ok and factory_ok and regulation_ok),
    }
    return work, checks


def run_strategy_comparison(
    models: pd.DataFrame,
    response: pd.DataFrame,
    budget_mclp: float = TOTAL_BUDGET_MCLP,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    detail_frames, check_rows, scenario_frames = [], [], []
    base_allocations = heuristic_allocations(models, budget_mclp)
    optimized, _ = optimize_budget(models, response, budget_mclp, demand_factor=1.0)
    base_allocations["Optimized"] = optimized
    for strategy, allocation in base_allocations.items():
        detail, checks = evaluate_allocation(strategy, allocation, models, response, demand_factor=1.0)
        detail_frames.append(detail)
        check_rows.append(checks)

    for scenario, factor in SCENARIOS.items():
        allocation, _ = optimize_budget(models, response, budget_mclp, demand_factor=factor)
        detail, checks = evaluate_allocation("Optimized", allocation, models, response, demand_factor=factor)
        detail.insert(1, "scenario", scenario)
        detail.insert(2, "demand_factor", factor)
        scenario_frames.append(detail)
        checks = {"scenario": scenario, **checks}
        check_rows.append(checks)

    detail_all = pd.concat(detail_frames, ignore_index=True)
    comparison = detail_all.groupby("strategy", as_index=False).agg(
        total_budget_mclp=("allocated_mclp", "sum"),
        expected_sales=("expected_sales", "sum"),
        expected_incremental_sales=("expected_incremental_sales", "sum"),
        expected_revenue_mclp=("expected_revenue_mclp", "sum"),
        expected_contribution_mclp=("expected_contribution_mclp", "sum"),
        expected_incremental_profit_mclp=("expected_incremental_profit_mclp", "sum"),
        inventory_depletion_units=("inventory_depletion_units", "sum"),
    )
    return detail_all, comparison, pd.DataFrame(check_rows), pd.concat(scenario_frames, ignore_index=True)


SENSITIVITY_FACTORS = (0.8, 0.9, 1.0, 1.1, 1.2)


def _scale_response(response: pd.DataFrame, factor: float, model: str = None) -> pd.DataFrame:
    scaled = response.copy()
    mask = slice(None) if model is None else scaled["model"] == model
    scaled.loc[mask, "incremental_units_per_mclp"] *= factor
    return scaled


def run_response_sensitivity(
    models: pd.DataFrame,
    response: pd.DataFrame,
    factors=SENSITIVITY_FACTORS,
    budget_mclp: float = TOTAL_BUDGET_MCLP,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Re-run the base-scenario comparison with the assumed response slopes scaled.

    Returns (global, by_model).  ``global`` scales every model's tier slopes by each
    factor and re-evaluates all four strategies.  ``by_model`` scales one allowed
    model's slopes by the smallest and largest factor while holding the others fixed,
    and reports the optimized result.
    """
    global_rows = []
    for factor in factors:
        scaled = _scale_response(response, factor)
        detail, comparison, checks, _ = run_strategy_comparison(models, scaled, budget_mclp)
        base_checks = checks.loc[checks["scenario"].isna()].set_index("strategy")["all_constraints_ok"]
        optimized = detail.loc[detail["strategy"] == "Optimized"].set_index("model")["allocated_mclp"]
        for row in comparison.itertuples():
            global_rows.append({
                "slope_factor": factor,
                "strategy": row.strategy,
                "total_budget_mclp": row.total_budget_mclp,
                "expected_incremental_sales": row.expected_incremental_sales,
                "expected_incremental_profit_mclp": row.expected_incremental_profit_mclp,
                "all_constraints_ok": bool(base_checks[row.strategy]),
                **({f"alloc_{m}": optimized[m] for m in optimized.index} if row.strategy == "Optimized" else {}),
            })

    model_rows = []
    for model in models.loc[models["regulatory_allowed"], "model"]:
        for factor in (min(factors), max(factors)):
            allocation, _ = optimize_budget(models, _scale_response(response, factor, model), budget_mclp)
            detail, checks = evaluate_allocation(
                "Optimized", allocation, models, _scale_response(response, factor, model)
            )
            model_rows.append({
                "model": model,
                "slope_factor": factor,
                "total_budget_mclp": float(detail["allocated_mclp"].sum()),
                "model_budget_mclp": float(detail.loc[detail["model"] == model, "allocated_mclp"].sum()),
                "expected_incremental_profit_mclp": float(detail["expected_incremental_profit_mclp"].sum()),
                "all_constraints_ok": bool(checks["all_constraints_ok"]),
            })
    return pd.DataFrame(global_rows), pd.DataFrame(model_rows)
