# Gold Price Prediction (Prophet + XGBoost)

Gold price forecasting with Facebook/Meta `Prophet` and XGBoost, using statistically screened external regressors. The original **monthly** Prophet model lives in [`model_building/prophet.ipynb`](model_building/prophet.ipynb); the newer **daily** Prophet model lives in [`model_building/gold_daily_prophet.ipynb`](model_building/gold_daily_prophet.ipynb); the daily XGBoost challenger lives in [`model_building/gold_daily_xgboost.ipynb`](model_building/gold_daily_xgboost.ipynb). All notebooks are fully executed, with outputs and plots.

## Pipeline (6 steps)

1. **Regression screening** — Pearson correlation matrices on gold price *levels*, 1-month and 12-month log changes (levels alone give spuriously inflated correlations on trending series).
2. **Preprocessing** (pandas) — all series normalized to a month-start grid; daily crude oil resampled to monthly means; QA cross-check between the two gold sources.
3. **Prophet fit** with `add_regressor` for the selected external factors.
4. **Historical data plot.**
5. **Prediction** — 6 months ahead; future regressor values come from auxiliary Prophet models.
6. **Forecast plot** with 95% uncertainty band + component decomposition.

## Regressor screening results

| Variable | r (levels) | r (12-month change) | Verdict |
|---|---|---|---|
| Brent crude | **+0.58** | +0.15 | **Kept** |
| US Dollar Index | +0.05 | **−0.51** | **Kept** — hidden by levels, strong in 12-month changes |
| WTI crude | +0.55 | +0.12 | Dropped — 0.99 collinear with Brent |
| Fed funds rate | −0.18 | −0.19 | Dropped — below both thresholds |

## Validation (24-month holdout, 2024-10 → 2026-09)

| cps | log y | MAPE | RMSE |
|---|---|---|---|
| **0.5** | **yes** | **25.8%** | **$1,194** |
| 0.2 | yes | 26.3% | $1,209 |
| 0.05 | no | 34.0% | $1,514 |

The holdout window deliberately covers the historic 2024–2026 gold surge — a stress test, not typical error.

## 6-month forecast snapshot (from the executed notebook)

| Month | yhat | 95% band |
|---|---|---|
| 2026-10 | $3,980 | 3,487 – 4,483 |
| 2027-03 | $4,309 | 3,780 – 4,916 |

## Daily model (new)

The daily notebook uses the Kitco USD price from `gold_data_new.xlsx` as the target and a business-day panel of external regressors.

- **Cleaning:** `Dates`-only parsing; invalid/out-of-range dates are dropped without manual repair. Duplicate dates are resolved against the local median, weekends are removed, and one outlier is flagged. The cleaned target has **2,652** observations and **406** missing business days.
- **WGC:** no WGC cross-check in the daily pipeline. The WGC workbook is retained only for the legacy monthly model.
- **Oil screening:** daily Brent/WTI levels and 20-day changes are tested against `|r| >= 0.5` (levels) and `|r| >= 0.3` (20-day changes). Neither passes, so oil is excluded from the final model.
- **Validation:** 24-month holdout and four expanding 12-month walk-forward folds.
- **Final regressors:** US 10-year yield, US Dollar Index, Fed funds rate, and CPI year-over-year, all lagged one business day.
- **Model:** log target, multiplicative seasonality, yearly seasonality, no weekly seasonality, `changepoint_prior_scale=0.05`.
- **Forecast anchoring:** the final path is shifted to the last observed price using the model's log residual at the forecast origin.

| Metric | Value |
|---|---|
| 24-month holdout MAPE | **6.35%** |
| 24-month holdout RMSE | **$420.1** |
| Walk-forward mean MAPE | **5.99%** |
| Walk-forward mean RMSE | **$242.7** |
| Forecast origin | **2026-10-26: $4,015.80** |
| 6-month forecast (2027-04-20) | **$4,634.57** (95% band $3,866.92–$5,616.03) |

Outputs are in `model_building/output_daily/` (cleaned target, feature panel, model comparison, holdout predictions, forecast, and PNG figures).

## Daily XGBoost challenger (new)

The XGBoost notebook tests whether a gradient-boosted tree model can beat the daily Prophet model on the same data and validation windows. It does not replace Prophet in production.

- **Target:** average daily log return over each horizon, `log(P[t+h]/P[t]) / h`.
- **Trend bridge:** subtract a 252-business-day trailing drift, capped at `±0.0008` per day (about ±20% annualised), before fitting. Forecasts are reconstructed from the last observed price. This prevents the tree model from extrapolating an unbounded price trend.
- **Primary design:** direct multi-horizon model. Each training row is one `(origin, horizon)` pair; `horizon` is a feature and the forward return is the label.
- **Diagnostic design:** recursive one-step model, used to measure error accumulation.
- **Features:** 75 selected features — gold technicals, US 10-year yield, dollar index, fed funds, CPI, real yield, gold-minus-macro spreads, origin/target calendar terms, and known-future macro levels/changes. No raw gold price or raw time index. No random split; training targets are strictly before each validation cutoff.
- **Feature-set choice:** selected by walk-forward MAPE, not by the outer holdout.
- **Future macro:** historical values at the target date for evaluation; `future_regressors_daily.csv` for the final forecast.
- **Oil:** Brent and WTI were tested as separate origin-feature ablations. Neither improved walk-forward MAPE, so the published model is oil-free.
- **Hyperparameters:** conservative baseline (`400` trees, depth `3`, learning rate `0.03`, subsample/colsample `0.8`, `min_child_weight=20`, L2 `10`, L1 `1`). Tuning was not performed on the outer holdout.

### Feature-set comparison

| Feature set | Features | 24m holdout MAPE | 12m walk-forward mean MAPE |
|---|---:|---:|---:|
| **Core + future macro (selected)** | **75** | **7.40%** | **11.20%** |
| Core + future macro + Brent | 80 | 7.68% | 11.36% |
| Core origin | 65 | 8.27% | 11.52% |
| Core + future macro + market | 88 | 8.05% | 11.53% |
| Core + future macro + WTI | 80 | 7.53% | 11.55% |

### Validation against Prophet and random walk

| Model | 24m holdout MAPE | Holdout RMSE | 12m WF mean MAPE | WF RMSE | WF MAE |
|---|---:|---:|---:|---:|---:|
| **Prophet selected 4-regressor** | **6.35%** | **$420.09** | **5.99%** | **$242.70** | **$197.49** |
| XGBoost direct | 7.40% | $453.79 | 11.20% | $409.37 | $328.06 |
| XGBoost recursive | 10.72% | $626.75 | 8.48% | $304.57 | $251.06 |
| Random walk | 26.05% | $1,314.37 | 11.45% | $424.66 | $340.31 |

XGBoost direct is closer to Prophet on the 24-month holdout but materially worse across the walk-forward folds, where it is only marginally better than a random walk on MAPE. The recursive variant is better than direct on walk-forward but worse on the holdout. Prophet remains the preferred model.

### 6-month forecast snapshot

Forecast origin: **2026-10-26, $4,015.80**. Final forecast date: **2027-04-20**.

| Horizon | Date | Point forecast | 95% band | Change from origin |
|---|---|---|---:|---:|
| 1 business day | 2026-10-27 | $4,018 | $3,920 – $4,119 | +0.06% |
| 1 month | 2026-11-24 | $4,087 | $3,650 – $4,577 | +1.78% |
| 3 months | 2027-01-21 | $4,206 | $3,457 – $5,117 | +4.73% |
| 6 months | 2027-04-20 | $4,411 | $3,343 – $5,821 | +9.85% |

### Interval calibration

The notebook first fits independent 2.5%, 50%, and 97.5% XGBoost quantile models. They under-covered the holdout (**52.4%** empirical coverage, **21.8%** mean width), so they are retained only as diagnostics and are not presented as a 95% interval.

The published band is a lognormal interval scaled by 63-day realised gold volatility:

`lower/upper = yhat * exp(±1.96 * sigma_63d * sqrt(h))`

This band covered **99.4%** of the 24-month holdout with a mean width of **52.9%** of price — deliberately conservative. The forecast-origin 63-day volatility is **1.26% per day**.

Outputs are in `model_building/output_daily_xgb/` (`xgb_model_comparison.csv`, `xgb_feature_set_comparison.csv`, `xgb_parameter_sensitivity.csv`, `xgb_holdout_predictions.csv`, `xgb_walk_forward_metrics.csv`, `xgb_forecast_summary.csv`, `xgb_forecast_daily.csv`, `xgb_interval_coverage.csv`, feature importance, and PNG figures).

## Data (`data-sources/`)

- `gold_data_new.xlsx` — daily Kitco USD target used by the new daily model (2015-02-05 → 2026-10-26 after cleaning)
- `Gold_price_averages_in_a range_of_currencies_since_1978.xlsx` — monthly WGC averages; legacy monthly target only, no longer used for daily QA (1978-01 → 2026-09, no gaps)
- `data_gpr_daily_recent.xls` — daily geopolitical risk index (requires `xlrd`)
- `brent_historical.csv` / `wti_historical.csv` — daily historical crude prices used by the daily model
- `DCOILWTICO.csv` / `DCOILBRENTEU.csv` — legacy monthly-model inputs (FRED daily series resampled to monthly)
- `FEDFUNDS.csv` — Fed funds effective rate (monthly)
- `US Dollar Index Historical Data.csv` — Investing.com monthly (US-style MM/DD dates, newest first)
- `Europe_Brent_Spot_Price_FOB.csv` — duplicate EIA Brent series (dropped)
- `monthly_csv.csv` / `annual_csv.csv` — older gold series, used for QA cross-check only

## How to run

```bash
python -m venv .venv
.venv/bin/pip install prophet pandas openpyxl xlrd matplotlib seaborn jupyter ipykernel nbconvert xgboost scikit-learn
.venv/bin/jupyter nbconvert --to notebook --execute --inplace model_building/prophet.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace model_building/gold_daily_prophet.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace model_building/gold_daily_xgboost.ipynb
```

Paths inside the notebooks are resolved relative to the repository root, so they run from any machine with this layout.

## Repo layout

- `data-sources/` — raw inputs (kept for reproducibility)
- `model_building/prophet.ipynb` — monthly full pipeline + documentation
- `model_building/gold_daily_prophet.ipynb` — daily Prophet full pipeline + documentation
- `model_building/gold_daily_xgboost.ipynb` — daily XGBoost challenger + validation
- `model_building/output/` — monthly exported figures
- `model_building/output_daily/` — daily Prophet exported figures, metrics, and forecast CSVs
- `model_building/output_daily_xgb/` — daily XGBoost exported metrics, forecasts, and figures
