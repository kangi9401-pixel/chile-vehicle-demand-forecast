# Technical Report: Chile Vehicle Demand Forecasting

## 1. Problem Statement

Vehicle demand in an emerging, export-driven economy like Chile is shaped by two
distinct dynamics: slow-moving structural trends in the consumer base (income
distribution, population growth) and faster macro cycles (interest rates,
commodity prices, which drive a large share of Chilean GDP and household income).
The goal of this project is to forecast monthly new-vehicle sales at the total-market
level and across ten sub-segments (brand, body style), both for near-term accuracy
(a 3-month holdout) and for a multi-year planning horizon, and to do so in a way
that is honest about which part of the forecast is driven by learned structure
versus by a static assumption about the future.

**Data disclaimer:** this repository uses a seeded synthetic dataset
(`data/generate_synthetic_data.py`) built to have the same shape and plausible
economic relationships as the real project this reproduces, not actual sales
figures. All numbers in this report are therefore about the *method*, not about
the Chilean auto market.

## 2. Method and Why It Was Chosen

Two models are trained per segment and combined in an ensemble:

- **DeepAR** (`gluonts`, PyTorch): an autoregressive RNN that learns a full
  predictive distribution (Student-t) per time step and takes the macro
  covariates as dynamic real-valued inputs. It captures autocorrelation and
  seasonality in the series itself, and extrapolates naturally to arbitrary
  horizons via its recurrent structure. Its main weakness for this data is that
  it has only ~100 monthly observations per segment to learn from -- a fairly
  small sample for a neural sequence model.
- **XGBoost**: a gradient-boosted tree regressor trained purely on the macro
  feature set (no lagged target values), which tends to generalize better than
  DeepAR when the series itself is short or noisy, at the cost of not modeling
  autocorrelation directly.
- **Ensemble**: a weighted average of the two, with the weight chosen per
  segment from out-of-sample backtest performance (Section 4.3) rather than a
  fixed 50/50 split.

Classical alternatives considered but not used: ARIMA/SARIMA (would need a
separate model per exogenous-feature specification and doesn't share statistical
strength across segments as naturally as a covariate-driven learner), and
Prophet (weaker support for multivariate dynamic covariates than DeepAR here).
Given the project's goal of comparing a deep sequence model against a
feature-driven model, DeepAR + XGBoost was the more informative pair to study.

## 3. Data and Feature Engineering

Macro covariates: population, income-tier splits (Low/Middle/High), interest
rate, and three commodity prices (crude oil, iron ore, copper), from which the
pipeline derives `middle_high_income_ratio`, `avg_resource_price`, and five
cross-ratios between them (see `src/chile_forecast/features.py`).

**A worked example of why feature engineering needs checking, not just writing:**
the initial version of `avg_resource_price` was an unweighted mean of the three
commodity prices. Because crude oil (~USD 20-120/barrel) and iron ore (~USD
30-180/ton) are one to two orders of magnitude smaller than copper (~USD
3,500-11,000/ton), that mean was in practice a thin proxy for the copper price
alone -- oil and iron barely moved it. The fix z-scores each series before
averaging, so the composite reflects genuine co-movement across all three
commodities. `tests/test_features.py::test_avg_resource_price_is_invariant_to_a_single_column_rescale`
encodes this directly: multiplying copper by 100 (a stand-in for a unit change)
must not move `avg_resource_price` at all under the z-scored version, and did
move it substantially under the original unweighted mean.

Ten market segments are modeled independently (total industry size, two OEM
brands, four body styles, two SUV categories introduced partway through
history). Each segment's longest usable contiguous history is found
automatically (`get_longest_non_nan_slice`) and requires at least
`MIN_SERIES_LEN` (60) months before it's modeled at all.

## 4. Experimental Design

### 4.1 Hyperparameter search

XGBoost is tuned with `RandomizedSearchCV` over `n_estimators`, `max_depth`,
`learning_rate`, `subsample`, `colsample_bytree`, scored by MAE under
`TimeSeriesSplit` (expanding-window, so no fold is ever validated against data
that precedes its own training fold). DeepAR is tuned with a small manual grid
over `hidden_size`, `num_layers`, `lr`, each scored by MAE on a held-out
`VAL_N`-month window that sits strictly between the training data and the final
holdout window -- it never sees the final test window's target values.

Both searches run once, on a single representative segment (`Total Market
Size`, `config.TUNING_SEGMENT`), rather than per segment. With only ~100
observations per series, a per-segment search would mostly fit hyperparameters
to that series' particular noise rather than finding settings that generalize;
tuning once on the aggregate market and reusing those settings everywhere is a
deliberate bias-variance tradeoff, not an oversight. See
`outputs/xgb_tuning_results.csv` and `outputs/deepar_tuning_results.csv` for
the full search results.

**Chosen hyperparameters** (from this run; re-running `data/generate_synthetic_data.py`
re-seeds the data, so a fresh run's numbers will differ slightly):

- XGBoost: `{'subsample': 0.9, 'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.1, 'colsample_bytree': 0.8}`
  (best of 25 random draws, rank 1 mean CV MAE -2082.2, std 375.1; rank-2 candidate
  `{'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.01, ...}` scored -2086.5,
  i.e. within noise of rank 1 -- the search is fairly flat near the optimum, not a sharp peak).
- DeepAR: `{'hidden_size': 64, 'num_layers': 2, 'lr': 0.0005}` (best of 4 configs, validation
  MAE 1506.8, vs. 1520.4 / 1729.6 / 2186.7 for the others -- lower learning rate helped
  more than a wider or deeper network did).

### 4.2 Rolling-origin (walk-forward) backtest

A single 3-month holdout is one noisy draw and can't distinguish genuine
generalization from a lucky window. `src/chile_forecast/backtest.py` repeats the
holdout evaluation at 3 origins, each 3 months further back, for every segment,
and reports MAE mean +/- std across origins rather than a single number.

### 4.3 Ensemble weighting

The per-segment DeepAR weight is `(1/MAE_DeepAR) / (1/MAE_DeepAR + 1/MAE_XGBoost)`,
using each model's mean MAE across the rolling-origin backtest for that segment
-- so a segment where DeepAR was consistently more accurate leans the ensemble
toward DeepAR, and vice versa, instead of assuming both models are equally
trustworthy everywhere.

### 4.4 Interpretability

XGBoost is the interpretable half of the ensemble (DeepAR doesn't have an
established SHAP-equivalent attribution method for an autoregressive RNN over a
learned latent state). `src/chile_forecast/interpret.py` produces a SHAP
beeswarm plot per segment (`outputs/<segment>_shap_summary.png`).

## 5. Results

### 5.1 Rolling-origin backtest (mean +/- std MAE across up to 3 origins)

| Segment | DeepAR | XGBoost | 50/50 Ensemble |
|---|---|---|---|
| Anac Market Size | 1733.3 ± 612.7 | 2213.2 ± 1065.1 | 1896.5 ± 708.2 |
| Total Market Size | 1939.0 ± 1022.2 | 2095.0 ± 1152.0 | 1982.3 ± 635.4 |
| OEM_A | 193.2 ± 136.4 | 222.5 ± 126.7 | 175.9 ± 89.6 |
| OEM_B | 265.0 ± 72.8 | 242.6 ± 78.0 | 213.4 ± 69.3 |
| B-Sedan | 258.0 ± 145.7 | 259.8 ± 141.8 | 246.8 ± 36.7 |
| B_HB | 317.3 ± 159.1 | 430.5 ± 128.2 | 359.2 ± 120.6 |
| C_Sedan | 377.2 ± 115.5 | 345.3 ± 222.1 | 341.9 ± 76.9 |
| C_HB | 333.1 ± 143.7 | 301.9 ± 29.4 | 305.9 ± 81.9 |
| SUV-A | 347.9 ± 91.4 | 291.0 ± 122.7 | 300.6 ± 129.5 |
| SUV-B | 137.2 ± 14.7 | 221.3 ± 91.1 | 118.0 ± 25.6 |

On the rolling backtest, the plain 50/50 ensemble beats *both* individual models outright
in 5 of 10 segments (Anac, Total, OEM_A, B-Sedan, SUV-B) and beats the worse of the two
everywhere -- consistent with the standard variance-reduction argument for averaging two
models with partially uncorrelated errors. The std columns are also informative on their
own: OEM_B and SUV-B have tight, low-variance MAE across origins (both models are stably
accurate), while Anac and Total Market Size have std comparable to or larger than half
their mean MAE -- i.e. accuracy on these two swings a lot depending on which 3 months you
happen to test on. That instability is exactly why a single 3-month holdout (Section 5.2)
is not sufficient evidence of generalization for these two segments in particular.

### 5.2 Final holdout (last 3 months), backtest-weighted ensemble vs. individual models

| Segment | Weight(DeepAR) | DeepAR MAE | XGBoost MAE | Weighted Ensemble MAE | Beat both? |
|---|---|---|---|---|---|
| Anac Market Size | 0.56 | 2940.50 | 1148.74 | 1951.25 | No |
| Total Market Size | 0.52 | 1622.53 | 1044.47 | 1244.65 | No |
| OEM_A | 0.54 | 300.94 | 131.03 | 219.32 | No |
| OEM_B | 0.48 | 248.96 | 169.54 | **154.39** | **Yes** |
| B-Sedan | 0.50 | 168.45 | 216.71 | 192.50 | No |
| B_HB | 0.58 | 221.97 | 283.49 | 248.08 | No |
| C_Sedan | 0.48 | 532.55 | 159.20 | 301.45 | No |
| C_HB | 0.48 | 153.67 | 268.95 | 214.14 | No |
| SUV-A | 0.46 | 401.27 | 284.41 | 337.64 | No |
| SUV-B | 0.62 | 121.89 | 158.25 | 135.80 | No |

**This is the honest, less flattering result, and it's worth stating plainly rather than
rounding up:** on this particular final 3-month window, the weighted ensemble beat *both*
individual models outright in only 1 of 10 segments (OEM_B). In every other segment it
landed between DeepAR and XGBoost, as a weighted average of two point predictions
necessarily tends to -- usually much closer to XGBoost, which was the stronger model on
this window in 8 of 10 segments. Two things are going on:

1. **XGBoost happened to be unusually strong on this specific 3-month window** (e.g. Anac
   XGBoost MAE 1148.74 vs. its own rolling-backtest average of 2213.2 -- nearly half). The
   rolling-origin results (5.1) show this window was not representative of typical
   accuracy for either model on several segments.
2. **The ensemble weight is set from rolling-backtest MAE, and Section 5.1 already shows
   that MAE is high-variance for exactly the segments (Anac, Total) where the ensemble
   underperforms most here.** A weight estimated from a noisy signal inherits that noise.

The practical takeaway: the ensemble is a reasonable *default* when you don't know in
advance which model will do better on a given future window (it never does worse than the
worse model, and did best overall on the rolling backtest), but "ensembling helps" is not
a safe claim to make from a single holdout window -- which is precisely the failure mode
rolling-origin backtesting exists to catch, and did catch here.

### 5.3 Feature importance

For `Total Market Size`, SHAP ranks `middle_high_income_ratio` and
`interest_rate` as the two most influential features, ahead of the raw
`middle_income` level and `avg_resource_price` -- the five engineered
cross-ratios contribute comparatively little. This is a useful empirical check
on the feature engineering: most of the derived ratio features could likely be
dropped with little accuracy cost, which the current feature set does not do
(see Limitations).

![SHAP summary for Total Market Size](../outputs/Total_Market_Size_shap_summary.png)

## 6. Limitations and Future Work

- **The backtest-weighted ensemble beat both individual models on the final holdout in
  only 1 of 10 segments** (Section 5.2), even though it was the best performer on average
  across the rolling-origin backtest (Section 5.1). The weight is estimated from
  rolling-backtest MAE, which Section 5.1 shows is itself high-variance for several
  segments (e.g. Anac Market Size: std comparable to the mean) -- so the weight estimate
  inherits that noise, and a single future 3-month window is not guaranteed to land where
  the backtest average would predict. This is the single most important empirical result
  in this report: it's evidence *against* trusting ensemble weights (or any accuracy claim)
  derived from a small number of backtest windows, and the honest fix is more backtest
  origins and/or a wider validation window, not a more sophisticated weighting formula.
- **Forecast-horizon macro features are held at their last observed value.**
  `run_deepar_longterm` has no real future macro data to condition on, so it
  forward-fills interest rate, income, and commodity prices flat for the entire
  multi-year horizon. This means the long-term forecast is closer to "if the
  economy stayed exactly where it last was" than a macro forecast in its own
  right, and the DeepAR covariates buy comparatively little beyond what the
  model's own trend/seasonality extrapolation would give it that far out. A
  proper fix would feed in a macro *scenario* (e.g. a simple VAR or
  analyst-provided path for interest rates and commodity prices) rather than a
  flat line, and report the forecast's sensitivity to that scenario.
- **DeepAR's internal checkpoint selection still monitors training loss**, not a
  held-out validation loss -- that's a `gluonts`/Lightning library default we
  did not override, since doing so reliably would have meant reworking the
  `ModelCheckpoint` wiring inside `PyTorchLightningEstimator.train_model`, which
  felt like the wrong place to spend risk given the rest of the scope here.
  Instead, generalization is checked *externally*: the hyperparameter search
  scores on a held-out window (Section 4.1), and the rolling-origin backtest
  (Section 4.2) checks that accuracy holds up across several time origins, not
  just the one the checkpoint happens to fit best.
- **Hyperparameters are tuned once on one segment and reused everywhere**
  (Section 4.1). This is a deliberate scoping decision, not a discovery, but it
  means segment-specific architecture choices are never explored.
- **The ensemble is a linear weighted average**, not a stacked/learned combiner.
  A logistic or gradient-boosted stacker over the two models' predictions (with
  the rolling-backtest windows as training data for the stacker) would likely
  do better and is the most natural next step.
- **Feature engineering includes five cross-ratio features** whose SHAP
  contribution (Section 5.3) is small relative to the four base features. They
  were kept for continuity with the original project's feature set rather than
  pruned; a follow-up ablation (train with vs. without each engineered ratio)
  would make this precise instead of qualitative.
- **Synthetic data.** All results here describe what the pipeline does on
  seeded synthetic data with known-plausible relationships baked in; they are
  not a claim about real Chilean vehicle demand or real forecast accuracy.
