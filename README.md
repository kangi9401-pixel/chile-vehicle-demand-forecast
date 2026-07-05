# Chile Vehicle Demand Forecast

A probabilistic demand-forecasting pipeline for the Chilean automotive market, combining a
**DeepAR** neural forecaster with **XGBoost** gradient boosting in an ensemble, driven by
macroeconomic covariates (income distribution, interest rates, commodity prices).

> **Data disclaimer:** This repo ships a synthetic dataset (`data/generate_synthetic_data.py`)
> that mimics the shape and relationships of the original real-world project — it is seeded
> random data, not actual sales or market figures from any company. This project reproduces
> the *methodology* from work I did professionally, rebuilt from scratch on public-style data
> for a portfolio-safe demo.

## Why this approach

Vehicle demand is driven by both **structural macro trends** (a growing middle/high-income
population buys more cars; higher interest rates suppress financed purchases; commodity
prices proxy for the health of Chile's export-driven economy) and **short-term momentum** in
the sales series itself. A single model struggles to capture both:

- **DeepAR** (`gluonts`, PyTorch-based) is an autoregressive RNN that learns a full predictive
  distribution per time series and naturally incorporates the macro features as dynamic
  real-valued covariates, extrapolated years into the future.
- **XGBoost** is trained on the same macro feature set as a purely feature-driven regressor,
  which tends to generalize better when the historical series itself is short or noisy.
- The **ensemble** (simple average) is backtested against both individual models on a holdout
  window and consistently reduces MAE relative to either alone.

## Architecture

```
data/generate_synthetic_data.py   Synthetic macro + segment sales dataset (seeded, reproducible)
src/chile_forecast/
  config.py         Paths, date ranges, segment list, model hyperparameters
  features.py       Longest-valid-slice utility, time features, macro feature engineering
  metrics.py         MAE / RMSE / MAPE / R2 fit metrics
  deepar_model.py    DeepAR holdout backtest + long-term (multi-year) forecast
  xgb_model.py       XGBoost holdout regressor
  ensemble.py        DeepAR + XGBoost ensemble averaging and holdout summary table
  visualize.py       Observed + forecast plot with Bollinger-band uncertainty context
  pipeline.py        Orchestrates the full run across every market segment
run_pipeline.py       CLI entry point
tests/                Unit tests for the pure utility functions
```

Each of the ten market segments (total industry size, per-brand sales, body-style segments
like sedans/hatchbacks/SUVs) is modeled independently, since each has a different history
length and demand pattern — segments only get modeled once they clear a minimum history
length (`MIN_SERIES_LEN`), and the pipeline finds each segment's longest usable contiguous
history automatically (handling segments introduced partway through the dataset, e.g. newer
SUV categories).

## Methodology

1. **Feature engineering** — derive income-tier ratios, commodity-price composites, and their
   cross-ratios with interest rates as macro covariates.
2. **Holdout backtest** — train DeepAR and XGBoost on all but the last 3 months per segment,
   predict the held-out months, and compare MAE for DeepAR vs. XGBoost vs. the ensemble.
3. **Long-term forecast** — retrain DeepAR on the full observed history and forecast out to a
   multi-year horizon, with 80% prediction intervals.
4. **Visualization** — observed + forecast series plotted alongside DeepAR's prediction
   interval and a rolling Bollinger Band (±2σ) on the observed data for additional volatility
   context.

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python data/generate_synthetic_data.py   # builds data/synthetic_chile_data.csv
python run_pipeline.py                   # trains models, writes outputs/
pytest tests/                            # unit tests for the utility functions
```

Outputs (git-ignored, generated locally):
- `outputs/<segment>_forecast.png` — per-segment observed + forecast plot
- `outputs/holdout_deepar_xgb_ensemble_compare.xlsx` — per-segment holdout comparison table

## Sample output

![Sample forecast](docs/sample_forecast.png)

## Tech stack

Python, PyTorch, GluonTS (DeepAR), XGBoost, scikit-learn, pandas, matplotlib.
