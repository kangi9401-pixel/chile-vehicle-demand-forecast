"""Publication-oriented charts for forecast and allocation results."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 130,
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })


def _save(fig: plt.Figure, output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    path = output_dir / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def create_decision_charts(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    allocation_detail: pd.DataFrame,
    comparison: pd.DataFrame,
    scenario_detail: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    _style()
    paths = []

    target = "B-Sedan" if "B-Sedan" in set(predictions["target"]) else predictions["target"].iloc[0]
    latest = predictions.loc[
        (predictions["target"] == target)
        & (predictions["origin_number"] == predictions["origin_number"].max())
        & (predictions["horizon"] == 12)
    ]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    actual = latest.drop_duplicates("date").sort_values("date")
    ax.plot(actual["date"], actual["actual"], color="black", marker="o", lw=2.4, label="Actual")
    for i, (model, group) in enumerate(latest.groupby("model")):
        group = group.sort_values("date")
        ax.plot(group["date"], group["predicted"], color=COLORS[i % len(COLORS)], linestyle=LINESTYLES[i % len(LINESTYLES)], lw=1.7, label=model)
    ax.set_title(f"Models diverge across the latest 12-month backtest — {target}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Synthetic vehicle sales (units)")
    ax.legend(ncol=2)
    fig.autofmt_xdate()
    paths.append(_save(fig, output_dir, "forecast_actual_vs_predicted.png"))

    perf = summary.loc[summary["horizon"] == 12].groupby("model", as_index=False)["WAPE_mean"].mean().sort_values("WAPE_mean")
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(perf["model"], perf["WAPE_mean"], color=COLORS[0])
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_title("Simple baselines remain a demanding forecasting benchmark")
    ax.set_xlabel("Mean WAPE across synthetic vehicle classes (%) — lower is better")
    ax.set_ylabel("Model")
    ax.set_xlim(0, max(perf["WAPE_mean"].max() * 1.18, 1))
    paths.append(_save(fig, output_dir, "model_performance_comparison.png"))

    horizon_perf = summary.groupby(["model", "horizon"], as_index=False)["WAPE_mean"].mean()
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for i, (model, group) in enumerate(horizon_perf.groupby("model")):
        ax.plot(group["horizon"], group["WAPE_mean"], marker="o", color=COLORS[i % len(COLORS)], linestyle=LINESTYLES[i % len(LINESTYLES)], label=model)
    ax.set_xticks(sorted(horizon_perf["horizon"].unique()))
    ax.set_title("Forecast error changes materially with decision horizon")
    ax.set_xlabel("Forecast horizon (months)")
    ax.set_ylabel("Mean WAPE across classes (%)")
    ax.legend(ncol=2)
    paths.append(_save(fig, output_dir, "horizon_performance.png"))

    best_model = perf.iloc[0]["model"]
    dist = metrics.loc[(metrics["horizon"] == 12) & (metrics["model"] == best_model)]
    labels = sorted(dist["target"].unique())
    values = [dist.loc[dist["target"] == label, "WAPE"].to_numpy() for label in labels]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(values, tick_labels=labels, patch_artist=True, boxprops={"facecolor": COLORS[0], "alpha": 0.65})
    ax.set_title(f"{best_model} error varies across vehicle classes and origins")
    ax.set_xlabel("Synthetic vehicle class")
    ax.set_ylabel("12-month WAPE by origin (%)")
    paths.append(_save(fig, output_dir, "target_error_distribution.png"))

    opt = allocation_detail.loc[allocation_detail["strategy"] == "Optimized"].sort_values("allocated_mclp")
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(opt["model"], opt["allocated_mclp"], color=COLORS[2])
    ax.bar_label(bars, fmt="%.1f", padding=3)
    ax.set_title("Profit-aware allocation concentrates budget where marginal return is highest")
    ax.set_xlabel("Promotion budget (million CLP, simulated)")
    ax.set_ylabel("Synthetic vehicle model")
    paths.append(_save(fig, output_dir, "optimized_budget_by_model.png"))

    comp = comparison.sort_values("expected_incremental_profit_mclp")
    fig, ax = plt.subplots(figsize=(8.5, 5))
    colors = [COLORS[0] if s != "Optimized" else COLORS[1] for s in comp["strategy"]]
    bars = ax.barh(comp["strategy"], comp["expected_incremental_profit_mclp"], color=colors)
    ax.bar_label(bars, fmt="%.0f", padding=3)
    ax.set_title("Optimized spending improves net incremental profit in this simulation")
    ax.set_xlabel("Expected incremental profit after promotion (million CLP, simulated)")
    ax.set_ylabel("Allocation strategy")
    paths.append(_save(fig, output_dir, "strategy_profit_comparison.png"))

    pivot = scenario_detail.pivot_table(index="scenario", columns="model", values="allocated_mclp", aggfunc="sum").reindex(["downside", "base", "upside"])
    fig, ax = plt.subplots(figsize=(10, 5.3))
    x = np.arange(len(pivot.index))
    width = 0.8 / max(len(pivot.columns), 1)
    for i, model in enumerate(pivot.columns):
        ax.bar(x - 0.4 + width / 2 + i * width, pivot[model], width, label=model, color=COLORS[i % len(COLORS)])
    ax.set_xticks(x, [s.title() for s in pivot.index])
    ax.set_title("Optimal promotion mix responds to ±15% demand scenarios")
    ax.set_xlabel("Demand scenario (assumption-based)")
    ax.set_ylabel("Promotion budget (million CLP, simulated)")
    ax.legend(ncol=2)
    paths.append(_save(fig, output_dir, "scenario_budget_allocations.png"))
    return paths


def create_extension_charts(
    calibration: pd.DataFrame,
    sensitivity: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Quantile calibration and response-slope sensitivity charts."""
    _style()
    paths = []

    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle=":", label="Perfect calibration")
    for i, (model, group) in enumerate(calibration.groupby("model")):
        group = group.sort_values("nominal_level")
        ax.plot(group["nominal_level"], group["observed_share_below"], marker="o", ms=4,
                color=COLORS[i % len(COLORS)], linestyle=LINESTYLES[i % len(LINESTYLES)], label=model)
    horizon = int(calibration["horizon"].iloc[0])
    ax.set_title(f"Quantile calibration over {horizon}-month windows")
    ax.set_xlabel("Nominal quantile level")
    ax.set_ylabel("Share of actuals at or below the forecast quantile")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    paths.append(_save(fig, output_dir, "interval_calibration.png"))

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for i, (strategy, group) in enumerate(sensitivity.groupby("strategy")):
        group = group.sort_values("slope_factor")
        ax.plot(group["slope_factor"], group["expected_incremental_profit_mclp"], marker="o",
                color=COLORS[i % len(COLORS)], linestyle=LINESTYLES[i % len(LINESTYLES)], label=strategy)
    ax.set_title("Allocation ranking under ±20% changes to the assumed promotion response")
    ax.set_xlabel("Multiplier on every incremental-units-per-MCLP slope")
    ax.set_ylabel("Net incremental profit (million CLP, simulated)")
    ax.legend()
    paths.append(_save(fig, output_dir, "optimization_sensitivity.png"))
    return paths
