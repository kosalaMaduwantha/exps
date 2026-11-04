# Gold Price Forecast API and Streamlit Dashboard — Implementation Plan

## Implementation status — 2026-11-04

- Implemented the modular `model_lib` extraction, typed API contracts, versioned artifact publication, FastAPI endpoints, and Streamlit/Plotly dashboard.
- Migrated the six daily notebooks and the three sibling notebooks to direct library helper imports; preserved existing notebook outputs and analyst edits. Experimental helpers use explicit context arguments and remain diagnostic-only recipes.
- Added `07_publish_dashboard_artifacts.ipynb` and published the existing four model bundles and forecast snapshots locally without refitting. Runtime publications are excluded from source control.
- Training, CV, selection, holdout evaluation, auxiliary regressor fitting, and final refit remain notebook workflows; the CLI now exposes only non-training helpers.
- Added cell-level traceability and a module/contract guide under `docs/`; setup and serving commands are in `README.md`.
- Verification includes existing time-safety tests, publication/API/client/Plotly/Streamlit tests, all-family saved-bundle forecast parity, notebook validation, and static typing. Full training/CV/refit and new five-horizon holdout evaluation were not executed.

## Objective and project boundary

Implement the model library, API, Streamlit presentation layer, and notebook-driven model-artifact workflow inside the `gold_forecasting/` project root. The first dashboard version consumes API responses to display test-evaluation charts, future-forecast charts, and an interactive spot-prediction form with model-specific regressor values. Use Plotly for dashboard charts. Keep model training, cross-validation, and model/parameter selection as notebook-only workflows for now, calling reusable `model_lib` functions. A separate offline training command, script, or automated runner is outside the current scope. Use this project's implementation as the validated starting point. Review all three notebooks in the sibling `gold_prediction/model_building` directory for the complete preprocessing, model-building, evaluation, forecasting, and reporting capabilities the library must support. Use the material-cost forecaster only as a reference for organizing directories; reuse gold-specific implementation code rather than importing its material-cost modeling logic.

The reusable gold logic currently lives in `gold_common_utils.py`, `gold_model_utils.py`, and `gold_plot_utils.py`. Use these implementations as the extraction and behavior-parity reference for a dedicated `model_lib` package with clear ownership of preprocessing, model training, cross-validation, evaluation, forecasting, artifacts, and reports. Update all notebooks, offline scripts, and API services to import and call the new library's public functions directly as part of the migration. Do not retain the old utility modules as compatibility facades or maintain parallel implementations of common logic; remove them after their consumers have been migrated and parity has been verified.

## Existing data and model contract

- Treat `config.json` and the gold forecasting README as authoritative. The target is workbook column 10, Kitco USD; its quotation unit is not verified.
- Preserve the configured Monday-Friday calendar and horizons 5, 21, 63, 126, and 252 weekday-grid steps. Do not describe this as a verified gold-market holiday calendar.
- Expose the existing `load_prepared()` behavior through the public `model_lib` preprocessing interface and use it in notebooks, offline scripts, and serving, preserving the config snapshot, raw-source hash, panel/feature alignment, and validation checks. Do not duplicate the feature pipeline or load it through a legacy utility module.
- Preserve the documented as-of/release-delay assumptions, observed-target rules, fold-local preprocessing, fixed-origin semantics, and model-specific feature and artifact formats.
- The validation report records forecast-only refits on the new horizons. No CV, holdout scoring, hyperparameter reselection, or model ranking was performed for that run. Present API values as model forecasts, not validated accuracy claims.

## Model library structure and responsibilities

Create the following structure within `gold_forecasting`. The principal directories follow the requested `preprocessing`, `model_training`, and `evaluation` organization. Additional modules separate workflows that have distinct responsibilities and consumers.

```text
gold_forecasting/
  model_lib/
    __init__.py                 # Stable public functions shared by notebooks, CLI, and API.
    contracts.py                # Prepared data, bundles, forecasts, and evaluation results.
    preprocessing/
      loaders.py                # Source readers and monthly/daily alignment.
      validation.py             # Target QA, duplicates, missingness, and source audit.
      availability.py           # Release-aware as-of joins and staleness rules.
      features.py               # Causal features and model-specific input preparation.
      screening.py              # Correlations, lag analysis, and feature-group screening.
    model_training/
      trainer.py                # Shared fit functions invoked by notebook workflows.
      prophet.py                # Prophet training and supported recipe options.
      xgboost.py                # Direct-horizon models and optional challenger designs.
      recurrent.py              # Shared GRU/LSTM sequences, scaling, and epoch selection.
      selection.py              # Inner-validation parameter and feature selection.
    cross_validation/
      splitters.py              # Chronological folds, horizon purging, and eligibility.
      walk_forward.py           # Fresh model/preprocessing per fold and checkpoints.
    evaluation/
      metrics.py                # Point metrics, baseline skill, and horizon summaries.
      holdout.py                # Locked-configuration final holdout evaluation.
      comparison.py             # Common-target model and baseline comparison.
      diagnostics.py            # Residuals, sensitivity, ablations, and error accumulation.
      intervals.py              # Optional interval diagnostics and coverage assessment.
    forecasting/
      predictor.py              # Shared loaded-bundle inference and price reconstruction.
      regressors.py             # Future regressor paths and availability validation.
    artifacts/
      serialization.py          # Model-specific save/load and metadata validation.
      registry.py               # Immutable versions, promotion, and rollback.
    reporting/
      exports.py                # Predictions, metrics, feature dictionaries, and audits.
      plots.py                  # Optional EDA, forecast, component, and diagnostic plots.
  run_pipeline.py                # Existing non-training workflows calling model_lib; no training/CV/selection entry point.
  gold_api/                      # FastAPI schemas/routes and thin serving service.
    schemas.py                   # Typed requests, responses, rows, and error contracts.
    services/                    # Bundle inference and published dashboard-data readers.
    routes/                      # Model, forecast, history, and evaluation endpoints.
  gold_dashboard/
    app.py                       # Streamlit entry point and view navigation.
    api_client.py                # Typed HTTP client, response parsing, timeouts, and errors.
    charts.py                    # Pure Plotly figure builders from typed API data.
    views/
      test_evaluation.py         # Published test results and metrics.
      future_forecast.py         # Observed history and future forecast points.
      spot_prediction.py         # Metadata-driven regressor form and prediction result.
```

- `preprocessing` prepares sources and causal feature candidates; learned imputation/scaling belongs to each fitted model pipeline and is fitted only on that fold's training data.
- `model_training` builds models; `cross_validation` controls chronology and independent fits; `evaluation` scores frozen predictions. Avoid a single wrapper that mixes these responsibilities.
- Training, cross-validation, model/parameter selection, and final refit are initiated and orchestrated only in notebooks for this phase. Their reusable algorithms remain in `model_lib`; CLI scripts and API services must not expose or invoke these workflows.
- `forecasting` is shared by notebooks, offline forecasts, evaluation, and FastAPI, ensuring matching feature order, transforms, horizon semantics, and inverse transformations.
- `artifacts` owns persistence rather than embedding file-writing logic in every trainer. Bundle metadata records recipe ID, model family, feature schema, calendar/frequency, horizons, configuration, training cutoff, preprocessing state, and evaluation provenance.
- `reporting` operates on returned tables/results and is optional for API serving. Plotting packages must not be required merely to load the API.
- `gold_dashboard` is an API-only presentation client: it must not read workbooks, registries, bundles, or notebook exports directly, import training implementations, or calculate model predictions/evaluation metrics. The API returns chart-ready data; dashboard helpers only transform it for presentation. Keep HTTP access, view logic, and Plotly figure construction separate.
- Keep abstractions small: typed prepared-data, trained-bundle, forecast-result, and evaluation-report contracts plus a model adapter interface for fit/predict/save/load. Preserve model-specific parameters and transformations instead of forcing all models into identical internals.
- Publish the shared workflow functions through documented `model_lib` package/subpackage interfaces. Notebooks must use these same preprocessing, training, cross-validation, evaluation, forecasting, artifact, and reporting functions as the CLI and API wherever the workflow overlaps. Keep only configuration, orchestration calls, and explanatory narrative in notebooks; move reusable algorithms into the library.
- Use named configuration recipes for the validated daily four-model workflow, legacy monthly Prophet, daily Prophet experiments, and daily XGBoost challenger experiments. Do not silently combine their targets, cleaning policies, feature definitions, or training objectives.

## Notebook workflow coverage

Create a traceability table during extraction mapping each notebook cell or logical step to a library entry point, configuration recipe, output artifact, and verification scenario. Inspect executable cells as well as markdown and existing exports. Migrate all six `gold_forecasting` notebooks and all three `gold_prediction/model_building` notebooks to short calls into the new `model_lib` public functions with explanatory narrative. Update their imports and execution setup to resolve the same package; do not route notebook calls through the old utility modules or copy library implementations into cells.

| Notebook/source | Capabilities to capture | Library ownership |
|---|---|---|
| `gold_prediction/model_building/prophet.ipynb` | Monthly source loading, month-start normalization, oil aggregation, applicable gold-source cross-checks, level/one-month/twelve-month change correlations, collinearity screening, target/log-scale and changepoint sensitivity, holdout evaluation, auxiliary regressor forecasts, six-month prediction, uncertainty/component plots | `preprocessing`, `model_training`, `evaluation`, `forecasting`, `reporting` |
| `gold_prediction/model_building/gold_daily_prophet.ipynb` | Gold-date and quote QA/audits, business-day alignment, source readers, release-aware macro alignment, missing/stale observations, oil screening, lag/cross-correlation analysis, regressor combinations, changepoint sensitivity, holdout and expanding walk-forward evaluation, residual diagnostics, anchored future regressor paths, forecast summaries and component/history plots | `preprocessing`, `cross_validation`, `model_training`, `evaluation`, `forecasting`, `reporting` |
| `gold_prediction/model_building/gold_daily_xgboost.ipynb` | Technical/macro/calendar features, horizon-aware training rows, cutoff purging, capped trailing-drift reconstruction, direct and recursive forecast designs, feature-group/oil ablations, parameter sensitivity, random-walk comparison, horizon error and recursive accumulation diagnostics, quantile/volatility-band coverage checks, feature importance and forecast exports | `preprocessing`, `model_training`, `cross_validation`, `evaluation`, `forecasting`, `reporting` |
| `gold_forecasting/01` through `06` notebooks and shared utilities | Validated cleaning/availability rules, fixed-horizon Prophet/XGBoost/GRU/LSTM pipelines, recurrent window eligibility and scaling, inner selection, development CV, holdout controls, final refit, bundle reload, common-target model comparison, and future forecast plots | All library subsystems; these implementations are the default daily recipe |

Coverage rules:

- Inventory all required steps before extraction; classify each as reused, extracted, adapted, or diagnostic-only. Record why a legacy step is adapted when its assumptions differ from the validated daily workflow.
- Preserve original source data and audit outputs. Keep legacy monthly gold-source QA separate from the daily target; use the validated daily cleaning policy by default rather than adopting conflicting duplicate/outlier handling automatically.
- Fit screening, feature-group selection, parameter selection, imputers, and scalers within training/inner-validation boundaries. Outer holdout results and full-series correlations must not drive deployed feature selection.
- The legacy XGBoost challenger uses a different direct/recursive design from the validated per-horizon XGBoost pipeline. Implement it as a named experimental recipe, with its own feature schema and reconstruction logic; do not replace the default adapter silently.
- Historical realized future macro values used by an experiment must be marked as an oracle/conditional diagnostic. Deployable evaluation must use origin-known inputs or regressor paths fitted through the origin. Save that distinction with every report.
- Support interval evaluation and plotting as optional capabilities. Report empirical coverage and width; do not label a band as calibrated or equivalent across recipes without supporting evaluation. Keep point-only forecasts valid for models with no interval support.
- Do not transfer the legacy notebooks' published metrics or preferred-model ranking to newly trained artifacts. Recompute results on matching targets, folds, horizons, and input-availability assumptions.

## Implementation changes

1. **Notebook-only training, cross-validation, selection, and artifact publication**
   - Keep training for individual models and the four-model workflow in the notebooks, using public `model_lib` functions. Do not add a separate offline training method, CLI stage, script, or automated runner in this phase. The API only loads and serves prepared data and published bundles.
   - Run existing development walk-forward CV from notebooks. Perform model/parameter selection there using library functions and development/inner-validation results; select final parameters using inner-validation results from the latest development fold, never the untouched holdout.
   - Perform final refit from notebooks with the model adapters extracted into `model_lib` on eligible observed history. Use library serialization functions to save immutable, versioned bundles containing model-specific preprocessing, feature columns, config, training cutoff, selected parameters, and CV metrics.
   - Publish notebook-produced bundles through explicit notebook calls to the artifact registry functions, atomically updating the manifest pointing each model ID to its active version. Keep old versions for rollback; exclude runtime registry and version directories from source control. Publication must not trigger training, CV, or selection.
   - Export test/holdout predictions, corresponding observed targets, baseline predictions, metrics by horizon, and future-forecast snapshots through typed library reporting functions for API consumption. Record report ID, model/bundle version, recipe, split, origin/target dates, horizons, data/config hashes, evaluation period, training cutoff, and regressor-availability assumptions. Link reports to their originating artifacts explicitly; a final-refit bundle must not be presented as the bundle that generated earlier holdout predictions.

2. **Reuse the model library**
   - Extract the existing gold utility functions into the `model_lib` structure above and preserve default behavior through parity checks against the original implementation. Make the new public library functions the direct dependency of every migrated consumer, including notebooks; do not add compatibility facades.
   - Add the notebook capabilities identified in the coverage table as shared functions and explicit recipe options. Avoid duplicated training/evaluation algorithms in notebooks, CLI scripts, or API services.
   - Replace legacy utility imports and inline reusable logic in all nine notebooks and relevant scripts with calls to the new library functions. Remove `gold_common_utils.py`, `gold_model_utils.py`, and `gold_plot_utils.py` once all consumers are migrated and verification passes.
   - Preserve Prophet origin-time and auxiliary-path modes, default XGBoost direct-horizon models, and GRU/LSTM windows and saved scalers. Make alternative notebook designs opt-in recipes with distinguishable artifact metadata.

3. **FastAPI serving**
   - Add a `gold_api` package with typed schemas, a service layer, and FastAPI entry point. Configure artifact root by environment variable with a project-local default.
   - Expose `GET /health`, `GET /ready`, `GET /models`, `GET /models/{model_id}`, `POST /forecasts/spot`, and `POST /forecasts`. Readiness checks prepared data and promoted artifacts.
   - Requests provide model ID, an observed `origin_date`, and the model's external feature values. The spot request also chooses one configured horizon; the range request returns all configured horizons. Return model/version, origin/target dates, horizon, predicted value, origin price, and configured target unit.
   - Gold lags, recurrent context, and calendar features come from the validated prepared panel. Model metadata and Swagger list the exact external feature names callers must supply.
   - Reject dates outside the prepared weekday calendar, origins without an observed target, unsupported horizons, missing/extra/non-finite regressors, inactive models, corrupt bundles, and incompatible metadata. Use 4xx for invalid requests and 503 for unavailable artifacts/data.
   - Derive target dates using the configured weekday-grid step. Do not imply arbitrary-date horizons or future realized regressor values are supported.
   - Add the dashboard data-serving endpoints and typed contracts specified below. Read published evaluation results without rerunning evaluation, training, CV, or selection. Serve live inference only through the existing forecast endpoints.

4. **Streamlit dashboard using Plotly**
   - Build three views: **Test evaluation**, **Future forecasts**, and **Spot prediction**. Fetch all model metadata, chart data, metrics, and predictions through the API client, with configurable API base URL and explicit connection/read timeouts.
   - Test evaluation: select model, published report, and horizon; display Plotly actual-versus-predicted and residual-over-time charts, plus saved point/baseline metrics and horizon summaries. Show the test period, sample count, recipe, model version, and evaluation provenance. Plot by target date and filter to one horizon/origin-series as needed to avoid merging overlapping forecasts. Only compare models on compatible common targets and splits.
   - Future forecasts: select model and published forecast snapshot, display observed history alongside forecast points at the configured horizon dates, and optionally show supported interval bands and the underlying values table. Label the forecast origin, training cutoff, version, and regressor-path assumptions. These are sparse configured-horizon predictions, not a continuous daily forecast; do not interpolate extra predictions or invent uncertainty bands. Show stale snapshot dates explicitly.
   - Spot prediction: obtain model-specific external regressor descriptors and eligible origins from the API; render a Streamlit form for model, observed origin, supported horizon, and required numeric regressor values. Validate finite values and completeness before submission to `POST /forecasts/spot`, then display the predicted value, origin price, origin/target dates, horizon, model/version, and units. Keep gold lags, recurrent context, and calendar features server-derived. If a model needs no external regressors, show an empty regressor form rather than fabricated inputs.
   - Use form submission so widget reruns do not repeatedly request inference. Retain the submitted inputs with their result in session state and invalidate or clearly label the result when inputs change. Cache read-only API data with bounded TTLs keyed by model version/report/snapshot and filters; allow refresh. Distinguish loading, empty reports, validation errors, API unavailability, and incompatible data with actionable messages.

5. **Dependencies, typing, and documentation**
   - Add FastAPI, Uvicorn, Streamlit, Plotly, and an HTTP client to this project's dependency setup without changing pinned modeling dependencies. Keep dashboard dependencies optional for API-only installations.
   - Document setup, preprocessing, notebook-driven training/CV/selection and artifact publication, API startup, dashboard startup (`streamlit run gold_dashboard/app.py`), API URL configuration, shared artifact storage, request semantics, and Swagger at `/docs`. State that standalone training automation is deferred.
   - Add module docstrings describing ownership, dependencies, and public interfaces across `model_lib`, `gold_api`, and `gold_dashboard`. Document public functions, methods, and route handlers with parameter types, return types/shapes, units, assumptions, side effects, and raised errors. Include usage examples for notebook workflows, API clients, and chart builders to make refactoring straightforward.
   - Annotate function arguments and return values throughout new and migrated code. Use dataclasses/typed records for library results, `Protocol` interfaces for model adapters, Pydantic request/response models at HTTP boundaries, and explicit optional/enum/literal types where applicable. Avoid unstructured dictionaries and `Any` except at documented third-party boundaries; document DataFrame columns, dtypes, index semantics, and array shapes where Python types alone are insufficient.
   - Give API client methods concrete parsed response types and chart builders explicit `plotly.graph_objects.Figure` returns. Establish a shared lightweight contract package or generated client schemas so the dashboard does not import API server/model runtime dependencies. Document schema versions, nullable fields, dates, error responses, and compatibility expectations; validate responses at the client boundary and include static type checking of the new typed modules in required checks.

## Dashboard API data contracts

Use typed JSON responses containing data rather than serialized figures or HTML. Include a schema version and relevant model version/report/snapshot identifiers. Serialize dates in ISO 8601 format and unavailable numeric values as `null`, never JSON NaN/Infinity. Validate date filters, horizon filters, and bounded pagination for time-series responses.

| Endpoint | Purpose and response contract |
|---|---|
| `GET /models` and `GET /models/{model_id}` | Active model/version, recipe, supported horizons, currency/target unit, training cutoff, interval support, available report/snapshot IDs, and external regressor descriptors: exact name, numeric type, required status, description, units when known, and actual model constraints. Do not invent default regressor values or bounds. |
| `GET /models/{model_id}/origins` | Paginated eligible observed forecast origins, optionally filtered by date range/horizon, validated against the selected bundle's context and feature requirements. Return a recommended latest eligible origin for the spot form. |
| `GET /history` | Filtered/paginated observed target dates and prices from the prepared panel, with data version and target-unit metadata. Preserve missingness and observed-target flags rather than plotting filled targets as observed. |
| `GET /models/{model_id}/evaluations` | Published report summaries with report ID, originating model version, recipe, split (including explicit test/holdout labels), period, horizons, sample counts, metric availability, and provenance. |
| `GET /models/{model_id}/evaluations/{report_id}` | Filtered/paginated prediction rows with origin date, target date, horizon, actual, prediction, optional baseline and interval bounds; saved metrics by horizon and matching population; evaluation provenance and totals. Metrics must declare their scope and must not silently change with pagination. |
| `GET /models/{model_id}/forecast-snapshots` | Published future-forecast summaries with snapshot ID, model version, origin, training cutoff, horizons, creation time, and input/regressor-path assumptions. |
| `GET /models/{model_id}/forecast-snapshots/{snapshot_id}` | Saved forecast points with target date, horizon, predicted value, optional supported interval bounds/level, origin price, and provenance. This reads notebook-published forecasts rather than executing a new forecast or refit. |
| `POST /forecasts/spot` | Typed model ID, expected model version, observed origin date, one supported horizon, and exact external regressor values; returns one forecast and resolved version/provenance. |
| `POST /forecasts` | Same inference contract for all configured horizons, for interactive forecast consumers where needed; no training or evaluation. |

Resolve active model metadata and inference consistently: include the expected model version in inference requests, and return a clear conflict response if promotion makes the form stale so the client can refresh metadata. Report/snapshot reads use their immutable originating version even after promotion. Return 404 for unknown report/snapshot IDs, an empty list for no published results, and 503 for unavailable/corrupt required data; document typed errors in OpenAPI. If no compatible test results exist, show an unavailable state rather than treating forecast-only refits as test evaluations.

## Verification and acceptance

- Require a completed notebook-to-library traceability table covering all three `gold_prediction` notebooks and the six `gold_forecasting` notebooks. Every necessary preparation, training, evaluation, forecast, and report step must have a library implementation or an explicitly documented diagnostic recipe.
- Verify extraction parity for the default daily recipe before enabling additional recipes: cleaned panel, causal features, fold membership, target purging, model inputs, predictions, and exports must match the existing implementation on the same data/configuration.
- Verify that all nine notebooks, the CLI, and API import shared workflow functions directly from `model_lib`, with no remaining imports of the three legacy utility modules or duplicated reusable workflow implementations. Check migrated notebook imports and lightweight execution paths; compare representative notebook and CLI/API inference results from the same library functions using matching inputs, recipes, and saved bundles. Verify that training, CV, model/parameter selection, and final refit are invoked only from notebooks, with no CLI or API entry points for these workflows.
- Exercise each subsystem independently: preprocessing audits and availability, fold-local fitting and screening, model serialization, holdout isolation, common-target metrics, regressor paths, and optional interval diagnostics. Compare monthly and daily recipes only within compatible data/evaluation contracts.
- Add lightweight unit tests for metadata/input lists, missing and unknown regressors, non-finite values, invalid/unobserved origin dates, unsupported horizons, target-date calculation, missing registry/artifacts, and HTTP error mapping. Mock bundles and predictions so API tests do not train models.
- Compare API output with existing gold forecast artifacts at matching origins; verify bundle reload and the prediction path for each model family.
- Run existing gold pipeline tests and compile the API modules. Run full CV/refit only as a separately requested notebook execution task; GRU/LSTM training is expensive. Record clearly that training, CV, and selection are notebook-only and never run at API startup or during requests.
- Swagger must describe typed requests/responses and model-specific regressors; serving must use only promoted versioned bundles; no training endpoint or training work may run inside a request.
- Test dashboard data endpoints against small published fixtures: report/version provenance, horizon/date filtering, pagination, missing reports, null interval fields, metric scope, and finite JSON serialization. Verify that reading dashboard data never invokes training, CV, selection, or evaluation computations.
- Verify all three Streamlit views with mocked API responses and a lightweight API/dashboard integration smoke test: Plotly trace dates/values match served rows, spot form fields follow model metadata, submitted regressors reach inference unchanged, and stale-version conflicts, empty data, and connection errors render correctly. The dashboard must not access model/data files directly.
- Check typed client parsing and run the selected static type checker on the new contract, API, client, and chart modules. Verify documented argument/return contracts and API examples match the implemented schemas.

## Assumptions

- Prophet, XGBoost, GRU, and LSTM are the v1 model set.
- Model training, cross-validation, model/parameter selection, and final refit remain notebook-only for now, backed by reusable `model_lib` functions. The API consumes notebook-published artifacts; standalone offline training automation is deferred.
- Streamlit is the v1 presentation layer and Plotly is its chart library. Dashboard data and predictions come exclusively from API responses. Test charts require published test/holdout reports; future charts initially display notebook-published forecast snapshots. Spot predictions use live inference against promoted bundles.
- The request date is a historical forecast origin with an observed gold price; target dates are only the configured horizons.
- External inputs are exact model feature names known at the origin. Gold-derived lag/sequence features come from prepared project history.
- Currency is USD, while the quotation unit remains unconfirmed. Automated source refresh and a verified holiday calendar are out of scope.
- `gold_prediction` provides workflow coverage and experimental recipes; `gold_forecasting` supplies the validated daily starting implementation. The material-cost project supplies only a directory-organization reference.

## Implementation sequence

1. Inventory notebook steps and produce the traceability table with existing function mappings and recipe differences.
2. Extract the validated gold utilities into `model_lib`, define its public functions, and establish behavior parity against the original implementation.
3. Add missing notebook workflows as explicit monthly/daily experimental recipes with shared reporting and provenance.
4. Migrate all nine notebooks and existing scripts to direct `model_lib` calls, verify their imports and representative workflows, and remove the legacy utility modules after all consumers are migrated. Do not introduce compatibility facades.
5. Complete notebook-driven training/CV/selection and final refit using public library functions, then verify notebook artifact saving, versioning, promotion, and rollback. Defer standalone training orchestration and CLI training stages.
6. Connect FastAPI serving to the library's public interfaces; verify shared inference behavior across notebooks, CLI, and API, and document function contracts and usage.
7. Define typed dashboard contracts and notebook exports, then implement evaluation, forecast-snapshot, history, and eligible-origin APIs with report/version provenance and documented errors.
8. Build the typed API client and Streamlit test-evaluation, future-forecast, and spot-prediction views using Plotly; verify API-only data access and model-driven regressor inputs.
9. Complete API/dashboard integration checks, static type checks, module/function documentation, and startup/configuration examples.
