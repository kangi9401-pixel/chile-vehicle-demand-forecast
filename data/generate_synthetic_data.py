"""Generates a synthetic Chilean macro/auto-market dataset for this project.

This data is entirely artificial (seeded random walks + plausible economic
relationships). It exists only to make the forecasting pipeline runnable and
reproducible end-to-end; it does not represent any real company's sales figures.
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
START = "2015-01-31"
END = "2025-05-31"
OUT_PATH = Path(__file__).resolve().parent / "synthetic_chile_data.csv"


def _bounded_random_walk(rng, n, start, drift, vol, low, high):
    walk = np.empty(n)
    walk[0] = start
    for i in range(1, n):
        walk[i] = np.clip(walk[i - 1] + drift + rng.normal(0, vol), low, high)
    return walk


def generate(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START, END, freq="ME")
    n = len(dates)
    month_of_year = dates.month.values
    t = np.arange(n)

    population = 18_500_000 + t * 4_500 + rng.normal(0, 5_000, n)
    high_share = 0.18 + 0.0015 * t / 12 + rng.normal(0, 0.004, n)
    middle_share = 0.42 + 0.0008 * t / 12 + rng.normal(0, 0.006, n)
    high_share = np.clip(high_share, 0.10, 0.35)
    middle_share = np.clip(middle_share, 0.30, 0.55)
    low_share = np.clip(1 - high_share - middle_share, 0.15, 0.6)

    high = population * high_share
    middle = population * middle_share
    low = population * low_share

    interest_rate = _bounded_random_walk(rng, n, start=3.5, drift=0.0, vol=0.35, low=0.5, high=11.5)
    interest_rate += 2.0 * np.sin(2 * np.pi * t / 48)

    crude_oil = _bounded_random_walk(rng, n, start=55, drift=0.05, vol=3.0, low=20, high=120)
    iron_ore = _bounded_random_walk(rng, n, start=70, drift=0.05, vol=4.0, low=30, high=180)
    copper = _bounded_random_walk(rng, n, start=6000, drift=6.0, vol=200.0, low=3500, high=11000)

    middle_high_income_ratio = (middle + high) / population
    avg_resource_price = (crude_oil + iron_ore + copper) / 3
    seasonality = 1 + 0.12 * np.sin(2 * np.pi * (month_of_year - 11) / 12)

    economic_index = (
        3.0 * middle_high_income_ratio
        - 0.02 * interest_rate
        + 0.00002 * (copper - copper.mean())
    )
    economic_index = economic_index - economic_index.min() + 0.5

    total_market_size = (
        28_000 * economic_index * seasonality
        + rng.normal(0, 900, n)
    )
    total_market_size = np.clip(total_market_size, 3_000, None)
    anac_market_size = total_market_size * (1 + rng.normal(0, 0.015, n))

    oem_a_share = np.clip(0.10 + 0.0006 * t / 12 + rng.normal(0, 0.006, n), 0.05, 0.18)
    oem_b_share = np.clip(0.11 - 0.0003 * t / 12 + rng.normal(0, 0.006, n), 0.05, 0.18)
    oem_a = total_market_size * oem_a_share
    oem_b = total_market_size * oem_b_share

    sedan_decline = np.clip(1 - 0.0015 * t, 0.35, 1.0)
    suv_growth = np.clip(1 + 0.006 * t, 1.0, 3.2)

    b_sedan = total_market_size * 0.16 * sedan_decline + rng.normal(0, 250, n)
    b_hb = total_market_size * 0.14 * sedan_decline + rng.normal(0, 250, n)
    c_sedan = total_market_size * 0.12 * sedan_decline + rng.normal(0, 250, n)
    c_hb = total_market_size * 0.10 * sedan_decline + rng.normal(0, 250, n)
    suv_a = total_market_size * 0.05 * suv_growth + rng.normal(0, 250, n)
    suv_b = total_market_size * 0.04 * suv_growth + rng.normal(0, 250, n)

    # New segments introduced partway through history (NaN before launch), so the
    # pipeline's "longest contiguous non-NaN slice" logic has real gaps to handle.
    suv_a[:24] = np.nan
    suv_b[:36] = np.nan

    df = pd.DataFrame({
        "Date": dates.strftime("%Y%m").astype(int),
        "Population": population.round(0),
        "Low": low.round(0),
        "Middle": middle.round(0),
        "High": high.round(0),
        "Interest Rate": interest_rate.round(3),
        "Crude_Oil_USD_per_Barrel": crude_oil.round(2),
        "Iron_Ore_USD_per_Ton": iron_ore.round(2),
        "Copper_USD_per_Ton": copper.round(2),
        "Anac Market Size": anac_market_size.round(1),
        "Total Market Size": total_market_size.round(1),
        "OEM_A": oem_a.round(1),
        "OEM_B": oem_b.round(1),
        "B-Sedan": np.clip(b_sedan, 0, None).round(1),
        "B_HB": np.clip(b_hb, 0, None).round(1),
        "C_Sedan": np.clip(c_sedan, 0, None).round(1),
        "C_HB": np.clip(c_hb, 0, None).round(1),
        "SUV-A": np.clip(suv_a, 0, None).round(1),
        "SUV-B": np.clip(suv_b, 0, None).round(1),
    })

    return df


if __name__ == "__main__":
    data = generate()
    data.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(data)} rows to {OUT_PATH}")
