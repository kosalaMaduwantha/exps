# Gold Price Prediction (Prophet + External Regressors)

Monthly gold price forecasting with Facebook/Meta `Prophet`, using statistically screened external regressors. All findings, QA checks and code live in [`model_building/prophet.ipynb`](model_building/prophet.ipynb) (fully executed, with outputs and plots).

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

## Data (`data-sources/`)

- `Gold_price_averages_in_a range_of_currencies_since_1978.xlsx` — target: WGC monthly averages, USD column (1978-01 → 2026-09, no gaps)
- `DCOILWTICO.csv` / `DCOILBRENTEU.csv` — FRED daily crude prices (WTI kept as candidate only)
- `FEDFUNDS.csv` — Fed funds effective rate (monthly)
- `US Dollar Index Historical Data.csv` — Investing.com monthly (US-style MM/DD dates, newest first)
- `Europe_Brent_Spot_Price_FOB.csv` — duplicate EIA Brent series (dropped)
- `monthly_csv.csv` / `annual_csv.csv` — older gold series, used for QA cross-check only

## How to run

```bash
python -m venv .venv
.venv/bin/pip install prophet pandas openpyxl matplotlib seaborn jupyter ipykernel nbconvert
.venv/bin/jupyter nbconvert --to notebook --execute --inplace model_building/prophet.ipynb
```

Paths inside the notebook are relative (`../data-sources`), so it runs from any machine with this layout.

## Repo layout

- `data-sources/` — raw inputs (kept for reproducibility)
- `model_building/prophet.ipynb` — full pipeline + documentation
- `model_building/output/` — exported PNG figures (history, regressors, correlation matrices, holdout validation, forecast, components)
