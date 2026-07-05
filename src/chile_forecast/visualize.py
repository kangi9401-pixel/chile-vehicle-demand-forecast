"""Observed + long-term forecast plotting with Bollinger-band uncertainty context."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chile_forecast.config import BOLL_K, BOLL_WINDOW, DATE_COL


def plot_observed_and_forecast(
    seg: str,
    series_obs: pd.DataFrame,
    longterm_result: dict,
    output_dir: Path,
) -> Path:
    obs_x = series_obs[DATE_COL]
    obs_y = series_obs[seg].values

    future_x = longterm_result["future_dates"]
    future_y = longterm_result["preds_mean"]
    future_low = longterm_result["preds_low"]
    future_up = longterm_result["preds_up"]

    all_x = pd.concat([obs_x.reset_index(drop=True), pd.Series(future_x)], ignore_index=True)
    all_y = np.concatenate([obs_y, future_y])
    all_low = np.concatenate([np.full(len(obs_x), np.nan), future_low])
    all_up = np.concatenate([np.full(len(obs_x), np.nan), future_up])

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(all_x, all_y, color="blue", lw=2, label="Observed + Forecast")
    ax.fill_between(all_x, all_low, all_up, color="orange", alpha=0.2, label="DeepAR 80% PI")

    roll_mean = series_obs[seg].rolling(window=BOLL_WINDOW).mean()
    roll_std = series_obs[seg].rolling(window=BOLL_WINDOW).std()
    ax.fill_between(
        series_obs[DATE_COL],
        roll_mean - BOLL_K * roll_std,
        roll_mean + BOLL_K * roll_std,
        color="gray", alpha=0.15, label=f"Bollinger Band ±{BOLL_K}σ",
    )

    ax.axvline(series_obs[DATE_COL].iloc[-1], ls="--", color="k", lw=1, label="Forecast Start")
    ax.set_title(f"{seg} – DeepAR Observed + Long-Term Forecast", fontsize=15)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{seg.replace(' ', '_').replace('/', '-')}_forecast.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
