# Validation performed

## Application implementation verification — 2026-11-04

The library/API/dashboard implementation was verified separately from model
accuracy experiments:

- **24 automated checks passed**, including the original ten time-safety checks,
  fixture-based artifact/API/client/Plotly tests, pure extracted-recipe tests,
  Streamlit AppTest interaction across all three views, and frozen reload parity
  for all four saved model families.
- `mypy` passed for **50 source modules** across `model_lib`, shared contracts,
  FastAPI, the typed client, and Streamlit views/chart builders.
- Ruff's Python correctness checks passed; application/library modules compiled.
- All six daily notebooks, the three sibling experimental notebooks, and the new
  publication notebook passed nbformat and executable-cell syntax validation.
- Notebook 07 executed successfully, copying existing model bundles and forecast
  snapshots into the git-ignored serving store and explicitly promoting them.
  It did not train, select, evaluate, or refit a model.
- The real published store returned HTTP 200 for health, readiness, model listing,
  and history. Spot inference returned HTTP 200 for Prophet, XGBoost, GRU, and LSTM
  with exact metadata-driven inputs. Each family exposed its saved snapshot.
- Actual Uvicorn and Streamlit processes passed HTTP startup checks for API
  health/readiness/Swagger and dashboard health/index; both were then shut down.
- Importing the API does not import `model_lib.model_training`; serving prediction
  uses frozen inference and native deserialization modules directly.

Verification used Python 3.12, the existing modeling environment, and the versions
in the new application dependency files. Tests were run in an isolated temporary
application environment. The actual saved-model integration command was:

```bash
GOLD_VERIFY_SAVED_MODELS=1 python -m pytest -q
python -m mypy
```

No full development CV, hyperparameter reselection, new five-horizon holdout
evaluation, or final model refit was performed for application validation.
The existing auxiliary-path unit test fits small synthetic regressors; API/UI
fixture checks and saved-bundle parity do not train. Original notebook output
cells remain historical; the nine research notebooks were not rerun end-to-end.
The dashboard therefore shows no published five-horizon test evaluation until
one is actually produced and published from a notebook.

## Passed checks

- Shared preprocessing executed on all supplied source files.
- Output: 3,058 weekday slots, 2,646 observed gold target prices, 412 unobserved
  slots and 75 candidate features before model-specific selection.
- Gold row audit: 2,646 retained, 147 exact-target duplicate rows excluded,
  28 rows belonging to conflicting duplicate groups quarantined, 17 weekend rows
  quarantined, 16 missing/out-of-bounds target rows excluded, 10 unresolved dates.
- Forecast-only refits executed for Prophet, XGBoost, GRU and LSTM using the
  existing locked hyperparameters and all available observed history.
- All four models produced finite forecasts for horizons 5, 21, 63, 126 and 252.
- Prophet future external-regressor paths were generated, anchored to the origin
  values and saved to `artifacts/prophet/future_regressors.csv`.
- Each model’s `future_forecast.csv` contains exactly five rows with the expected
  horizon labels, target dates, origin price and percent change from the origin.
- A combined future-forecast table was written to
  `artifacts/comparison/future_forecasts.csv`.
- Future forecast plots were generated for all four models.
- Ten focused automated tests passed: release timing, stale-data expiry, feature
  prefix invariance, target cutoff purging, observed sequence labels, final holdout
  separation, missing target preservation, negative WTI handling, date parsing and
  anchored future macro paths.
- Python modules compiled and all notebook JSON passed nbformat validation.

## Execution details

The current run used the command-line pipeline stages rather than cell-by-cell
notebook execution:

```bash
python run_pipeline.py --stage preprocess
python run_pipeline.py --stage forecast --model all
python -m unittest -v test_pipeline.py
```

Notebook sources were updated to match the new forecast-only behavior, but the
notebooks themselves were not re-executed cell by cell in this run. The delivered
notebooks remain ready to run in Jupyter.

Earlier `[1, 5, 20]` cross-validation, holdout and forecast artifacts were moved to
`artifacts/_archive_1_5_20/` and are not part of the current live outputs.

## Not performed

- Cross-validation or holdout scoring with the new five-horizon configuration.
- Hyperparameter reselection or model ranking.
- Evaluation of forecast accuracy on future observed prices.
- Verification of historical release vintages, publication timestamps, source price
  authenticity, gold quotation unit, or a gold-specific holiday calendar.
- Calibrated prediction intervals or statistical significance testing.

There is consequently no model ranking or forecasting-performance claim for the
new horizons. The saved values are conditional future forecasts, not validated
accuracy results.
