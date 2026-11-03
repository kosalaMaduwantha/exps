# Gold forecasting notebooks

Six readable notebooks implement shared preprocessing, Prophet, XGBoost, GRU,
LSTM, and a common comparison workflow. The original supplied data is included.
The default target is the workbook's Kitco USD column (zero-based index 10).
The source does not establish the quotation unit reliably; confirm it before use.

## Getting started

Use Python 3.11 or 3.12 in a fresh environment. From this folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name gold-forecasting --display-name "Gold forecasting"
jupyter lab
```

Choose the Gold forecasting kernel. On Windows, activate with
`.venv\Scripts\activate`; TensorFlow installation availability depends on platform.
`tensorflow-cpu` is the tested CPU distribution; compatible GPU environments can
use the corresponding TensorFlow distribution instead.

Run these notebooks in order:

1. `01_gold_common_preprocessing.ipynb`
2. Any or all of `02_gold_prophet.ipynb`, `03_gold_xgboost.ipynb`,
   `04_gold_gru.ipynb`, `05_gold_lstm.ipynb`
3. `06_gold_model_comparison.ipynb`

Keep both Python utility files and config.json beside the notebooks.
Each notebook explains its inputs, transformations, training and evaluation.
The utility functions contain the longer reusable logic and comments.

The alternative command-line workflow is:

```bash
python run_pipeline.py --stage preprocess
python run_pipeline.py --stage cv --model all
python run_pipeline.py --stage compare
python -m unittest -v test_pipeline.py
```

## Configuration and target choice

Edit `config.json` before notebook 01. Target column 10 is Kitco USD; other price
columns represent different local or institutional prices. Changing the target
requires updating the name, unit, bounds and possibly calendar assumptions.
The default bounds [500, 10000] are broad source-specific QA bounds, not estimated
statistical outlier thresholds. Do not apply them unchanged to local-currency gold.

The end date is the intended gold coverage endpoint, 2026-10-26. Clearly incorrect
future years are quarantined. Dates are recovered from the other workbook date
column when that provides an unambiguous in-range value. Conflicting valid dates
are quarantined. Source rows, raw dates, resolved dates and reasons are preserved.
Conflicting duplicate target quotes are excluded for review, not averaged or chosen
using later prices. Exact target duplicates retain the first row.

## Calendar and missing observations

The calendar is Monday–Friday, not a verified gold-exchange holiday calendar.
Weekend-dated gold rows are quarantined. Weekday gaps are reported as unobserved;
this project does not invent a holiday classification without a supplied calendar.
Forecast horizons [5, 21, 63, 126, 252] mean weekday-grid steps and are labeled
7 days, 1 month, 3 months, 6 months and 1 year. For example, a holiday weekday
still counts as one step. Missing target dates are never scored or filled as labels.

Gold input values can be forward-filled for at most five weekday steps; a missing
indicator is retained. Recurrent windows bridging six consecutive missing target
slots are excluded from training. Leading or other missing recurrent features use
training-fitted median imputation and feature standardization. Target log changes
are standardized separately. XGBoost handles missing features natively.

## External data and availability assumptions

Every external input is joined backward on its assumed availability date. Source
observation dates are retained indirectly through age features. No backward fill
or future interpolation is used. Raw source data is never overwritten.

- CPI: seasonally adjusted CPI-U CUSR0000SA0; monthly percentage change and annual
  percentage change are calculated on a complete monthly index. Availability is
  month-end plus 15 calendar days by default.
- Fed funds: monthly averages, month-end plus seven calendar days. The two rate
  files are identical and checked for agreement; only FEDFUNDS is used.
- DXY: monthly price candles, not daily observations. Complete months become
  available at month-end plus one calendar day. The latest potentially incomplete
  month is omitted. Five-/20-grid-step DXY changes are changes in the available
  stepped monthly series, not daily market returns.
- GPR: daily overall risk, acts, threats, and recomputed trailing seven-calendar-day
  averages. All are delayed by seven calendar days as an explicit conservative
  placeholder. This delay is not verified and may not reproduce the publication
  schedule. It preserves weekend information through calendar-time averaging
  and the as-of join. Event annotations and source-provided moving averages are
  not used as predictive features.
- Daily oil and 10-year yield quotes: one business-day delay. This avoids assuming
  US closing quotes were known when a local gold quote was recorded. Exact
  timestamps and release histories remain unknown.
- Staleness limits are source-specific and exported in availability_assumptions.csv.
- WTI can be negative: use price differences, not ordinary log returns.
- Brent spot and OHLCV histories are kept separately. The initial external-model
  feature group uses the explicitly labelled Brent spot series. The instrument
  behind the OHLCV histories is not independently verified; no roll adjustment
  or assumption that both Brent series are interchangeable is made.
- Annual gold data ending in 2019 is preserved in raw data but excluded from training.
- The Kaggle helper is preserved as an original source file but never executed.

These are historical snapshots, with no original release vintages. Delays do not
remove revision leakage or prove point-in-time availability. Backtest results must
be described as snapshot-based, conditional on the documented timing assumptions.
Use actual release calendars and vintage data for a verified real-time backtest.
No additional data is downloaded automatically by the notebooks.

## Model design

Prophet fits log prices with yearly and weekly seasonality. Its external variant
uses a small regularized regressor set. Fixed-origin development forecasts hold
each regressor at its last available origin value. The final future forecast keeps
the fitted coefficients frozen and instead feeds anchored auxiliary Prophet paths
fitted to each regressor’s daily as-of history through the origin. Missing
historical regressors use training medians. Constant regressors are omitted
within a fold. Target-derived regressors are excluded.

XGBoost fits one direct model per horizon to log(y[t+h]/y[t]), using lagged prices,
trailing returns/volatility and optional external inputs. Predictions reconstruct
the original price using the observed origin price. The trial grid selects tree
capacity and boosting rounds on inner validation; there is no outer early stopping.

GRU and LSTM use the same compact input groups, horizon outputs, trial budget and
sequence builder. Each predicts a vector of log price changes, reconstructed from
the observed origin quote. An inner chronological tail selects epochs; training
labels are purged from that tail. A freshly initialized network then refits all
eligible training examples for the selected number of epochs. Test windows may use
prior training history as context, but all window inputs end at the forecast origin.

## Evaluation contract

This deliverable is a forecast-only run. No cross-validation, holdout scoring or
model ranking was executed. The configuration still defines a valid future
evaluation schedule: three development folds with 360-slot validation blocks and
a 504-slot final holdout, enough room for the 252-step horizon. Existing locked
hyperparameters from the prior `[1, 5, 20]` run were reused without reselection.

Models refit once on all available observed history. Each future forecast starts
from the latest observed gold quote and uses only origin-time features, except
Prophet’s anchored auxiliary external paths described above. No calibrated
prediction intervals or significance claims are provided.

## Final refit and prediction

The current run uses the already locked hyperparameters and skips the holdout
stage. Run the final refit cell in each model notebook, or:

```bash
python run_pipeline.py --stage forecast --model all
```

Final refitting uses the locked configuration and all available observed history.
Saved forecasts contain the origin date, horizon, horizon label, target date,
origin price, forecast and percent change from the origin. Prophet also writes
`future_regressors.csv` with the anchored auxiliary external paths used for the
forecast. The final model directory includes input columns, training
configuration and cutoff, and all fitted transformations needed to reproduce
inference. `load_bundle` restores these artifacts. Loading joblib artifacts
should be limited to files you trust.

Earlier `[1, 5, 20]` CV, holdout and forecast artifacts were moved to
`artifacts/_archive_1_5_20/` and are not part of the current live outputs.

## Outputs and validation

- data/raw/: original supplied source files.
- data/processed/: shared panel, feature candidates, audit, dictionary, assumptions,
  fold schedule, exact preprocessing configuration and source manifest.
- artifacts/MODEL/: locked parameters, the refitted final model, future forecast,
  Prophet’s future external paths, and future forecast plots.
- artifacts/comparison/future_forecasts.csv: one combined table for all models.
- artifacts/_archive_1_5_20/: archived results from the previous `[1, 5, 20]` run.
- validation/: short workflow test results, explicitly separate from full experiments.
- VALIDATION.md: what was actually executed for this deliverable.

Notebook 01 is delivered with executed preprocessing outputs. The four model
notebooks retain their full default experiments, ready to run. Short workflow
checks are not evidence of forecasting quality. See VALIDATION.md for actual checks.

Changes to preprocessing configuration require rerunning notebook 01. Saved model
metadata retains its training configuration; new prediction horizons are rejected.
Changing input files likewise requires rebuilding processed data and model artifacts.
The manifest records input hashes for audit; this is a local research workflow,
not an automated data-update service.
