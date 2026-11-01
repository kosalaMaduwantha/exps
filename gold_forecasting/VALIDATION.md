# Validation performed

## Passed checks

- Shared preprocessing executed on all supplied source files.
- Output: 3,058 weekday slots, 2,646 observed gold target prices, 412 unobserved
  slots and 75 candidate features before model-specific selection.
- Gold row audit: 2,646 retained, 147 exact-target duplicate rows excluded,
  28 rows belonging to conflicting duplicate groups quarantined, 17 weekend rows
  quarantined, 16 missing/out-of-bounds target rows excluded, 10 unresolved dates.
- All six notebook workflows executed cell by cell with no remaining code errors.
- Model workflow tests used one outer development fold and one external-feature
  candidate, two maximum neural epochs and ten XGBoost trees. Prophet used its
  normal fitting procedure. These are installation/workflow tests, not final experiments.
- All four model adapters produced finite predictions for horizons 1, 5 and 20.
- All four saved model types were reloaded and reproduced predictions within 1e-5
  relative tolerance.
- The parameter-lock, holdout scoring, final refit and future forecast paths were
  tested for all four models on an early DEVELOPMENT subset, using a simulated
  holdout boundary. The actual final 252-slot holdout was not evaluated.
- Repeated holdout writes were rejected as intended.
- Nine focused automated tests passed: release timing, stale-data expiry, feature
  prefix invariance, target cutoff purging, observed sequence labels, final holdout
  separation, missing target preservation, negative WTI handling and date parsing.
- Python modules compiled and all notebook JSON passed nbformat validation.
- Preprocessing plot visually inspected.

## Execution details

Notebook code cells were executed in process with IPython and captured outputs.
This environment does not permit launching a socket-based Jupyter kernel, including
local IPC; normal nbclient kernel execution could therefore not be validated here.
The same code cells ran successfully without requiring sockets. The delivered
notebooks retain standard Python kernel metadata for use in Jupyter.

The main preprocessing notebook includes executed outputs. The main model and
comparison notebooks retain full experiment defaults and no misleading short-run
outputs. Executed shortened notebook copies and JSON summaries are under validation/.
Saved artifacts inside validation/lifecycle/ are TEST ONLY, trained on an early
subset, and must not be mistaken for final models.

## Not performed

- The full five-fold, three-candidate searches at up to 60 neural epochs.
- Evaluation of the actual final holdout.
- Final models trained on the complete current history.
- Verification of historical release vintages, publication timestamps, source price
  authenticity, gold quotation unit, or a gold-specific holiday calendar.
- Calibrated prediction intervals or statistical significance testing.

There is consequently no final model ranking or forecasting-performance claim.
Run the full development notebooks before interpreting comparative accuracy.
