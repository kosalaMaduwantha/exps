# Validation performed

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
