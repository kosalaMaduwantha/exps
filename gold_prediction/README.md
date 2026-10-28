# Gold Price Prediction (Prophet + External Regressors)

Gold price forecasting with Facebook/Meta `Prophet`, using statistically screened external regressors. The original **monthly** model lives in [`model_building/prophet.ipynb`](model_building/prophet.ipynb); the newer **daily** model lives in [`model_building/gold_daily_prophet.ipynb`](model_building/gold_daily_prophet.ipynb). Both notebooks are fully executed, with outputs and plots.

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

- **Cleaning:** robust date repair, duplicate-date resolution against the local median and WGC monthly average, weekend removal, and outlier flagging.
- **Validation:** 24-month holdout and four expanding 12-month walk-forward folds.
- **Final regressors:** US 10-year yield, US Dollar Index, Fed funds rate, and CPI year-over-year, all lagged one business day.
- **Model:** log target, multiplicative seasonality, yearly seasonality, no weekly seasonality, `changepoint_prior_scale=0.05`.
- **Forecast anchoring:** the final path is shifted to the last observed price using the model's log residual at the forecast origin.

| Metric | Value |
|---|---|
| 24-month holdout MAPE | **6.28%** |
| 24-month holdout RMSE | **$413.6** |
| Walk-forward mean MAPE | **6.04%** |
| Walk-forward mean RMSE | **$242.3** |
| Forecast origin | **2026-10-26: $4,015.80** |
| 6-month forecast (2027-04-20) | **$4,610.74** (95% band $3,915.20–$5,547.72) |

Outputs are in `model_building/output_daily/` (cleaned target, feature panel, model comparison, holdout predictions, forecast, and PNG figures).

## Data (`data-sources/`)

- `gold_data_new.xlsx` — daily Kitco USD target used by the new daily model (2015-02-05 → 2026-10-26 after cleaning)
- `Gold_price_averages_in_a range_of_currencies_since_1978.xlsx` — monthly WGC averages, used as the monthly target and as daily QA (1978-01 → 2026-09, no gaps)
- `data_gpr_daily_recent.xls` — daily geopolitical risk index (requires `xlrd`)
- `DCOILWTICO.csv` / `DCOILBRENTEU.csv` — FRED daily crude prices (WTI kept as candidate only)
- `FEDFUNDS.csv` — Fed funds effective rate (monthly)
- `US Dollar Index Historical Data.csv` — Investing.com monthly (US-style MM/DD dates, newest first)
- `Europe_Brent_Spot_Price_FOB.csv` — duplicate EIA Brent series (dropped)
- `monthly_csv.csv` / `annual_csv.csv` — older gold series, used for QA cross-check only

## How to run

```bash
python -m venv .venv
.venv/bin/pip install prophet pandas openpyxl xlrd matplotlib seaborn jupyter ipykernel nbconvert
.venv/bin/jupyter nbconvert --to notebook --execute --inplace model_building/prophet.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace model_building/gold_daily_prophet.ipynb
```

Paths inside the notebooks are resolved relative to the repository root, so they run from any machine with this layout.

## Repo layout

- `data-sources/` — raw inputs (kept for reproducibility)
- `model_building/prophet.ipynb` — monthly full pipeline + documentation
- `model_building/gold_daily_prophet.ipynb` — daily full pipeline + documentation
- `model_building/output/` — monthly exported figures
- `model_building/output_daily/` — daily exported figures, metrics, and forecast CSVs
