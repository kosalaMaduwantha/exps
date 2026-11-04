# Application architecture and refactoring guide

## Ownership and dependencies

| Package | Owns | Consumers |
|---|---|---|
| `model_lib.preprocessing` | Source QA, release-aware joins, causal features, freshness validation | Notebooks, non-training CLI, API |
| `model_lib.model_training` | Native model fitting, recurrent epoch selection, parameter trials | Notebooks only |
| `model_lib.cross_validation` | Split chronology, purging, nested walk-forward orchestration | Notebooks only |
| `model_lib.evaluation` | Frozen-prediction metrics, holdout orchestration, common-target comparison | Notebooks; CLI can compare existing saved results |
| `model_lib.forecasting.predictor` | Frozen inference and price reconstruction | Notebooks, API |
| `model_lib.forecasting.regressors` / `notebook` | Auxiliary-path fitting and final refit | Notebooks only |
| `model_lib.artifacts` | Native serialization, immutable publication, atomic promotion/rollback | Notebooks publish; API reads |
| `model_lib.reporting` | Table publication and optional Matplotlib notebook figures | Notebooks |
| `model_lib.recipes` | Named monthly/daily Prophet and XGBoost challenger helper algorithms | Original sibling notebooks; diagnostic-only |
| `gold_contracts.schemas` | Versioned Pydantic HTTP payloads | API and dashboard; no heavy model imports |
| `gold_api.service` | Freshness, native bundle cache, request-local scenarios, eligibility | FastAPI routes |
| `gold_dashboard.api_client` | HTTP transport, typed response parsing, bounded GET cache | Streamlit views |
| `gold_dashboard.charts` | Pure Plotly figure construction | Dashboard and tests |

Public package exports collect library functions; there are no legacy utility
facades or parallel algorithm implementations. Model-specific state is retained
in a typed `Bundle`; backend objects are the explicit heterogeneous typing
boundary. `PreparedData` and `ModelAdapter` document aligned data and adapter
interfaces. Functions retain family-specific parameters and transformations.

## Data shapes and semantics

- Prepared `panel`: unique ascending `DatetimeIndex` named `ds`, Monday–Friday.
  `y` is observed-only price; `target_observed` is 0/1. Missing targets are never
  filled as labels. Source ages reflect configured availability assumptions.
- `features`: same index as panel; causal candidate columns. Model column order
  comes from the saved bundle. Feature fill is not evidence of target observation.
- `folds`: `fold`, inclusive integer `cutoff`, inclusive integer `stop`.
- `predict_model`: origins are integer positions; output shape is
  `(len(origins), len(config['horizons']))`, reconstructed into price units.
- Recurrent inputs: `(batch, lookback, feature_count)`; saved imputers/scalers
  transform inputs, and saved target scalers invert predicted log changes.
- HTTP row types distinguish origins/targets/horizons. Nullable prices and
  intervals stay nullable; clients must not invent missing values or bands.
- Residual is `prediction - actual`, exported during notebook evaluation.

## Publication from notebooks

```python
from pathlib import Path
from model_lib.preprocessing import ROOT, load_prepared
from model_lib.artifacts.registry import ArtifactStore
from model_lib.reporting.exports import publish_saved_model, publish_future_table
import pandas as pd

load_prepared()  # Raises when source/config snapshots or feature alignment differ.
store = ArtifactStore(ROOT / 'artifacts/serving')
metadata = publish_saved_model(ROOT, store, 'xgboost', 'research-v2')
snapshot = publish_future_table(
    store, metadata, 'forecast-v2',
    pd.read_csv(ROOT / 'artifacts/xgboost/future_forecast.csv'),
    assumptions='Origin-time features; no realized future regressors.',
)
store.promote('xgboost', metadata.version)
```

`publish_saved_model` does not train or promote. It verifies native metadata and
prepared configuration, copies trusted files, and records checksums/data identity.
Model versions and JSON reports are immutable. Publication completes before an
atomic active-registry replacement. A single local publisher is supported.

For a real holdout report, publish the `holdout_model` using its own version and
call `publish_test_table` with that metadata and the corresponding predictions.
Do not substitute `final_model`: it has a later training cutoff. The publisher
rejects test origins at/before the training cutoff and checks metric populations
and residuals. Development selection uses the latest development fold's inner
boundary; holdout values do not choose parameters.

## HTTP and presentation examples

```python
from gold_contracts.schemas import SpotRequest
from gold_dashboard.api_client import ApiClient
from gold_dashboard.charts import evaluation_chart

with ApiClient('http://127.0.0.1:8000') as client:
    model = client.model('xgboost')
    origins = client.origins('xgboost')
    # Populate exact model.external_regressors names with origin-known scenario values.
    # For a gold-only model the mapping is empty.
    result = client.spot(SpotRequest(
        model_id=model.model_id, expected_version=model.version,
        origin_date=origins.latest, horizon=model.horizons[0], regressors={},
    ))
```

Chart builders take validated API records and return Plotly `Figure` objects.
For example `evaluation_chart(page.items, page.target_unit)` performs no network
access, model evaluation, or artifact reads. View code owns widget/session state;
client code owns HTTP errors and schema validation. A failed POST clears the old
result; an unsubmitted edit is explicitly labeled as not applied to the result.

## Safe refactoring workflow

1. Change a function in its owning module and its explicit argument/return types.
2. Update direct consumers and module/public docstrings together. For HTTP changes,
   update the shared contract and API/client examples; increment schema version
   for incompatible changes. Do not make independent API/client schema copies.
3. Run mypy, focused tests, and notebook cell syntax/import checks.
4. When inference behavior changes, run saved-bundle parity for all four families.
5. When notebook imports change, update generator templates without regenerating
   over analyst edits. Historical output cells are not proof of a rerun.

## Verification boundaries

Fixture-based API and Streamlit AppTest checks exercise all three views, Plotly
traces, exact regressor forwarding, form-only inference, input errors, pagination,
and publication provenance. Existing causality tests and actual saved-bundle
reload parity check the default daily pipeline. These checks do not establish
forecast accuracy. No full CV, holdout run, or refit is part of application setup.
