# Technical Report: Chile Vehicle Demand Forecasting

## 1. Problem Statement

Monthly new-vehicle demand in an export-driven economy like Chile mixes slow structural
drivers (income distribution, population), faster macro cycles (interest rates,
commodity prices) and strong seasonality. This project asks two questions:

1. Once preprocessing and model selection are free of look-ahead leakage, do DeepAR,
   XGBoost or their ensemble actually beat simple seasonal baselines, and how does that
   depend on the segment and the forecast horizon?
2. Given those forecasts, does a constraint-aware promotion-budget allocation produce
   more net incremental profit than allocation rules that follow demand alone?

**Data disclaimer:** all data is seeded synthetic data
(`data/generate_synthetic_data.py`, seed 42; promotion inputs in
`src/chile_forecast/promotion_response.py`). Every number here describes the *method* on
synthetic data, not the Chilean auto market or any company's results.

**Relationship to the original project:** the original internal model used a single
DeepAR+XGBoost combination. Model comparison, the seasonal-naive baseline,
rolling-origin backtesting and the macro covariates were added in this independent
rebuild on synthetic data.

## 2. Data

The generator produces 125 months (2015-01 to 2025-05) of population, low/middle/high
income counts, interest rate, crude oil / iron ore / copper prices and ten market and
segment sales series with seasonality, trend, macro effects and noise. The evaluation
uses the four body-style segments **B-Sedan, B_HB, SUV-A and SUV-B**. SUV-A and SUV-B
start with 24 and 36 missing months respectively, mimicking later product launches;
each series is evaluated on its observed span.

## 3. Method

### 3.1 Models

| Model | Description |
|---|---|
| `LastValue` | Repeats the last training observation |
| `SeasonalNaive` | Value from the same month one year earlier |
| `MovingAverage6` | Mean of the last six training months |
| `XGBoost` | 150 trees, seed 42, on leakage-safe macro, calendar (month sine/cosine) and trend features; no target lags |
| `DeepAR` | GluonTS/PyTorch, 24 hidden units, 2 layers, lr 1e-3, 2 epochs, with the same leakage-safe features as dynamic covariates |
| `LeakageSafeEnsemble` | `w·DeepAR + (1−w)·XGBoost`, `w = MAE_XGB / (MAE_DeepAR + MAE_XGB)` using 12-month MAE from strictly earlier origins only; `w = 0.5` at the first origin |

DeepAR is deliberately small: with roughly 100 monthly observations per series, extra
capacity adds runtime and overfitting risk rather than evidence. Python, NumPy and Torch
seeds are fixed at 42 and the Lightning trainer runs in deterministic mode; a test
checks that two DeepAR runs on the same input produce identical predictions.

### 3.2 Leakage-safe preprocessing

`LeakageSafePreprocessor` is fitted separately at every origin with training cutoff *t*:

- imputation medians and the mean/std used to z-score income, interest rate and each
  commodity price come only from rows with date ≤ *t*;
- evaluation rows are transformed with those training statistics;
- unknown future macro values are **not** read from the evaluation window — they are
  held at their last training value, and only calendar and trend features advance;
- ratio features that divided by a commodity z-score (which crosses zero and made the
  ratios explode) are not used.

`tests/test_decision_system.py::test_future_rows_cannot_change_training_preprocessing`
corrupts future macro values (copper ×1,000,000, interest rate −999) and asserts that
the training and forecast features do not change.

### 3.3 Backtest design

All models share **8 training cutoffs**, 3 months apart: 2022-08-31, 2022-11-30,
2023-02-28, 2023-05-31, 2023-08-31, 2023-11-30, 2024-02-29 and 2024-05-31. At each
cutoff every model makes one 12-month forecast, which is scored on its first **3, 6 and
12 months**, so every model sees exactly the same training data and target dates at
every horizon. This gives 4 segments × 8 origins × 3 horizons × 6 models = 576 metric
rows and 4,032 prediction rows (`forecast_metrics_by_origin.csv`,
`forecast_predictions.csv`).

### 3.4 Metrics

MAE, RMSE, **WAPE** (Σ|actual − forecast| / Σ|actual|, the headline metric), sMAPE, and
MASE scaled by the in-sample seasonal-naive error. Plain MAPE is not reported because it
is unstable near zero. Adjacent windows overlap by 9 months, so the spread across origins
describes variability but does not support independence-based significance tests.

## 4. Relationship to This Repository's Earlier Version

An earlier public version of this repository evaluated DeepAR, XGBoost and an
inverse-MAE ensemble on a single 3-month holdout plus 3 rolling origins, with MAE as the
only backtest metric and no baseline. An audit ([AUDIT_KO.md](AUDIT_KO.md)) found that
commodity prices were z-scored over the full 125 months before splitting, that
backtests used the evaluation window's actual macro values, and that ensemble weights
could include the origin being evaluated. Those earlier results are therefore not
comparable to the ones below and are not repeated here. The earlier workflow remains
available as `pipeline.run_legacy()` for traceability.

## 5. Forecast Results

All numbers are read from `outputs/decision_system/forecast_metrics_summary.csv` and
`forecast_improvement_vs_seasonal.csv`, produced by `python run_pipeline.py`.

### 5.1 Mean WAPE across segments

| Model | 3 months | 6 months | 12 months |
|---|---:|---:|---:|
| LeakageSafeEnsemble | **13.80%** | **14.48%** | **14.98%** |
| DeepAR | 14.70% | 15.29% | 15.91% |
| MovingAverage6 | 14.88% | 15.96% | 15.90% |
| XGBoost | 15.20% | 15.87% | 16.21% |
| SeasonalNaive | 18.38% | 18.46% | 18.47% |
| LastValue | 18.68% | 20.56% | 20.51% |

On the segment average, the ensemble's WAPE is 24.9%, 21.5% and 18.9% lower than
Seasonal Naive at 3, 6 and 12 months.

### 5.2 WAPE by segment and horizon

**3 months**

| Segment | LastValue | SeasonalNaive | MA6 | XGBoost | DeepAR | Ensemble |
|---|---:|---:|---:|---:|---:|---:|
| B-Sedan | 17.49 | 14.38 | 12.49 | 11.40 | 12.79 | **11.25** |
| B_HB | 17.06 | 16.04 | 12.65 | 13.14 | 11.99 | **11.79** |
| SUV-A | 15.84 | 16.61 | 15.48 | 14.59 | 14.30 | **13.90** |
| SUV-B | 24.32 | 26.48 | 18.91 | 21.68 | 19.73 | **18.27** |

**6 months**

| Segment | LastValue | SeasonalNaive | MA6 | XGBoost | DeepAR | Ensemble |
|---|---:|---:|---:|---:|---:|---:|
| B-Sedan | 19.40 | 15.31 | 14.24 | **11.60** | 13.79 | 11.91 |
| B_HB | 18.32 | 15.97 | 13.15 | 12.54 | 11.99 | **11.58** |
| SUV-A | 19.74 | 16.73 | 17.43 | 16.22 | 16.58 | **15.78** |
| SUV-B | 24.79 | 25.82 | 19.01 | 23.14 | 18.78 | **18.66** |

**12 months**

| Segment | LastValue | SeasonalNaive | MA6 | XGBoost | DeepAR | Ensemble |
|---|---:|---:|---:|---:|---:|---:|
| B-Sedan | 16.50 | 14.27 | 13.01 | **11.13** | 14.66 | 12.17 |
| B_HB | 18.41 | 16.94 | 14.03 | 13.14 | 13.15 | **12.59** |
| SUV-A | 20.94 | 17.08 | 17.00 | **15.85** | 18.28 | 16.32 |
| SUV-B | 26.20 | 25.58 | 19.56 | 24.70 | **17.57** | 18.85 |

**The ensemble was not always better, and the best model differed by segment and
horizon.** The ensemble is best in all four segments at 3 months and in three at 6
months. At 12 months it is best only for B_HB; XGBoost wins B-Sedan and SUV-A, and
DeepAR wins SUV-B. Relative to Seasonal Naive at 12 months, the best model improves
WAPE by 21.98% (B-Sedan), 25.68% (B_HB), 7.18% (SUV-A) and 31.33% (SUV-B). At the same
horizon, DeepAR is worse than Seasonal Naive for B-Sedan (14.66% vs 14.27%, −2.70%) and
SUV-A (18.28% vs 17.08%, −7.06%), and the ensemble is worse than XGBoost alone for those
two segments. On this data, a small neural model is not uniformly better than a
seasonal rule. An operational setup would need per-segment champion/challenger tracking
rather than one model for everything.

![Forecast horizon comparison](../outputs/decision_system/horizon_performance.png)

### 5.3 Feature importance (legacy workflow)

SHAP attribution is computed only in the legacy workflow (`run_legacy()`,
`interpret.py`). It uses that workflow's feature set, which still includes the
resource-price ratio features removed from the current pipeline, and it has not been
recomputed for the leakage-safe pipeline. For `Total Market Size`, it ranks
`middle_high_income_ratio` and `interest_rate` highest, followed by `middle_income` and
`avg_resource_price`. The engineered cross-ratios contribute comparatively little,
which is consistent with dropping the unstable ratio features. Treat this as a
qualitative check on the synthetic generator's built-in relationships, not as evidence
about real demand drivers.

![SHAP summary for Total Market Size (legacy workflow)](sample_shap_summary.png)

## 6. Promotion-Budget Optimization

### 6.1 Formulation

Five synthetic models (City Hatchback, Compact Sedan, Compact SUV, Family SUV, and a
Legacy Diesel SUV that regulation forbids selling) receive the latest leakage-safe
forecast as base demand, plus inventory, supply, factory capacity, lifecycle, price,
tariff-adjusted cost, unit margin, strategic weight and min/max budget. Units are
million Chilean pesos (MCLP).

Each model's spend is split into three tiers of 35, 45 and 55 MCLP with declining
incremental units per MCLP. For example, the Compact SUV tiers are 0.34, 0.22 and 0.11
units/MCLP. These slopes are assumptions, not estimates. With spend `x_ij`, slope
`r_ij`, unit margin `m_i`, lifecycle bonus `a_i` (phase-out 0.45, mature 0.10, growth 0)
and strategic weight `w_i`, `scipy.optimize.linprog` (HiGHS) maximizes

```
Σ_ij x_ij · ( r_ij · (m_i + a_i) · w_i − 1 )
```

subject to: total budget ≤ 300 MCLP; per-model min/max budgets; base demand plus
incremental units ≤ inventory plus supply; per-factory production ≤ capacity; and zero
spend on the regulated model. The reported net incremental profit is the unweighted
`Σ r_ij·x_ij·m_i − Σ x_ij`, so strategic weights do not inflate the financial figure.
Equal, forecast-share and prior-year-share heuristics are evaluated under the same
bounds and constraint checks.

### 6.2 Results

From `allocation_strategy_comparison.csv` and `constraint_checks.csv`:

| Strategy | Budget (MCLP) | Incremental units | Net incremental profit (MCLP) | All constraints met |
|---|---:|---:|---:|:---:|
| Equal | 300.00 | 71.50 | 138.27 | yes |
| Forecast share | 300.00 | 69.99 | 121.28 | yes |
| Prior-year share | 300.00 | 70.83 | 130.03 | yes |
| Optimized | **275.00** | 68.00 | **148.57** | yes |

The optimizer spends 275 MCLP (City Hatchback 35, Compact Sedan 80, Compact SUV 80,
Family SUV 80, Legacy Diesel SUV 0) and leaves 25 MCLP unused, because the remaining
tiers return less contribution than they cost under the assumed margins. It produces
3.5 fewer incremental units than equal allocation, but 7.45%, 22.5% and 14.26% more net
incremental profit than the equal, forecast-share and prior-year-share rules. So
maximizing volume and maximizing profit give different allocations.

Demand scenarios scale base demand by 0.85 / 1.00 / 1.15, capped at 96% of inventory
plus supply (`scenario_summary.csv`). The downside and base scenarios have the same
optimum (275 MCLP, 148.57 MCLP profit). In the upside scenario, base demand uses more
factory capacity, so the optimal spend *falls* to 184.11 MCLP (51.04 incremental units,
124.59 MCLP profit). Stronger demand does not automatically justify more promotion when
supply binds.

![Scenario allocation](../outputs/decision_system/scenario_budget_allocations.png)

## 7. Limitations and Future Work

- **Synthetic data and an assumed response function.** The results show that the
  method works, but have no external validity. The optimum is only as good as the
  assumed tier slopes; real use would need experimental or quasi-experimental estimates
  of promotion response.
- **Flat future macro covariates.** Holding macro values at their last observation
  avoids leakage but is not a realistic path. A proper backtest would use
  forecast vintages that were available at each origin.
- **Small DeepAR.** About 100 observations per series and 2 epochs make this a
  structural comparison, not a statement about neural forecasters in general.
- **Overlapping windows.** The 8 origins overlap, so errors are correlated across
  origins.
- **Deterministic scenarios.** Three point scenarios, no chance constraints, CVaR or
  distributionally robust optimization, and no forecast-value-added analysis of whether
  better accuracy changes the allocation.
- **Legacy SHAP.** Feature attribution (Section 5.3) was not recomputed for the
  leakage-safe feature set.
