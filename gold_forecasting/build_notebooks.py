"""Generate readable notebooks with explanations next to each workflow stage.

This module is a build script: rerunning it regenerates the notebook files in
the project directory. The notebook sources live here so that the actual
analyst-facing files stay small, focused, and free of duplicated boilerplate.
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent


def markdown(text):
    """Create one markdown notebook cell."""
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    """Create one executable notebook cell."""
    return nbf.v4.new_code_cell(text.strip())


def save(name, cells):
    """Validate and write one notebook to the project directory."""
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3',
            },
            'language_info': {'name': 'python', 'version': '3.11'},
        },
    )
    nbf.validate(notebook)
    nbf.write(notebook, ROOT / name)


# Common setup used at the top of every notebook.
setup = '''
from pathlib import Path
import os
# Limit CPU threads before importing optional numerical/model libraries.
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from gold_common_utils import ROOT, load_config
cfg = load_config()
pd.set_option('display.max_columns', 20)
print('Project:', ROOT)
print('Target:', cfg['target_name'], '| horizons:', cfg['horizons'])
'''

# ---------------------------------------------------------------------------
# Notebook 01: shared preprocessing.
# ---------------------------------------------------------------------------
save('01_gold_common_preprocessing.ipynb', [
markdown('''# Gold: shared preprocessing and causal features
Run this notebook first, from the project directory. It never changes the raw files.
The default target is column 10, Kitco USD. Confirm the quotation unit with the source.
If you change the target, also change `target_bounds`, the name and units in `config.json`.

Forecast origin t means the gold quote for t is already available. Horizons count Monday–Friday
slots, not verified exchange sessions. Missing target labels are never interpolated.

Historical release vintages are absent. CPI, Fed and GPR delays are explicit assumptions,
not proof that these snapshots were available historically. Results are therefore snapshot-based
backtests with assumed publication timing.'''),
code(setup),
markdown('## 1. Inspect raw source structure'),
code('''raw_dir = ROOT / 'data/raw'
source_inventory = pd.DataFrame([
    {'file': path.name, 'bytes': path.stat().st_size}
    for path in sorted(raw_dir.iterdir()) if path.is_file()
])
display(source_inventory)
display(pd.read_excel(raw_dir / 'gold_data_new.xlsx', header=None).head(8))'''),
markdown('''## 2. Clean, align, and create candidates
`preprocess` calls small documented functions in `gold_common_utils.py`.
Dates are reconciled only when one date column supplies an unambiguous in-range date.
Conflicting duplicate target prices and ambiguous dates are quarantined for review.
Exact target duplicates retain the first source row. Bounds reject obvious invalid values;
trailing robust flags are diagnostic and do not automatically remove genuine shocks.

External values enter through a backward availability-date join. DXY is monthly, WTI uses
absolute changes, and the redundant Fed file is checked but not duplicated as a feature.'''),
code('''from gold_common_utils import preprocess, make_folds, validate_panel
panel, features, cleaning_audit, availability = preprocess(cfg)
validate_panel(panel, features)
print('Calendar slots:', len(panel))
print('Observed target prices:', int(panel.target_observed.sum()))
print('Unobserved weekday targets:', int(panel.y.isna().sum()))
display(cleaning_audit.reason.replace('', 'retained').value_counts().rename('rows').to_frame())
display(availability)'''),
markdown('''## 3. Review exclusions and suspicious prices
No automatic year correction is made when both source dates are unreliable.
Use this audit to create a reviewed correction table before changing raw data.
Large-change flags remain part of the observed series.'''),
code('''display(cleaning_audit.loc[cleaning_audit.reason.ne('')].head(30))
display(panel.loc[panel.large_change_flag.eq(True), ['y', 'large_change_flag']].head(20))
missing = panel.loc[panel.y.isna()].copy()
missing['year'] = missing.index.year
display(missing.groupby('year').size().rename('unobserved_weekdays').to_frame())'''),
markdown('## 4. Plot target coverage and candidate features'),
code('''fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
panel.y.plot(ax=axes[0], title=f"Observed target: {cfg['target_name']}")
panel.target_observed.rolling(60).mean().plot(ax=axes[1], title='Observed fraction over 60 weekday slots')
axes[1].set_ylim(0, 1.05)
plt.tight_layout()
display(features.tail())
display(features.isna().mean().sort_values(ascending=False).head(15).rename('missing_fraction'))'''),
markdown('''## 5. Check causality by truncating history
Appending future rows must not change any already constructed feature.
Training-fitted scaling, imputation and model selection happen in the model notebooks.'''),
code('''from gold_common_utils import make_features
cut = len(panel) - cfg['holdout_steps']
prefix_features = make_features(panel.iloc[:cut], cfg)
pd.testing.assert_frame_equal(prefix_features, features.iloc[:cut])
print('Causal feature prefix check passed.')'''),
markdown('''## 6. Review cross-validation and holdout boundaries
Each outer fold selects its parameters using an earlier inner validation block.
The final holdout is excluded from tuning. Forecast origins are spaced by `origin_stride`.
A 20-step spacing reduces overlap for the longest initial horizon.'''),
code('''folds = make_folds(panel.index, cfg)
fold_view = folds.copy()
fold_view['train_through'] = [panel.index[i] for i in folds.cutoff]
fold_view['evaluate_through'] = [panel.index[i] for i in folds.stop]
display(fold_view)
print('Saved panel, candidates, feature dictionary, audits, assumptions and folds in data/processed/.')'''),
])

# ---------------------------------------------------------------------------
# Notebooks 02-05: model adapters with one extra inspection section each.
# ---------------------------------------------------------------------------
for number, kind, title in [
    (2, 'prophet', 'Prophet'),
    (3, 'xgboost', 'XGBoost'),
    (4, 'gru', 'GRU'),
    (5, 'lstm', 'LSTM'),
]:
    description = {
        'prophet': '''Prophet fits log gold prices with trend, weekly and yearly seasonality.
Gold-derived regressors are excluded. External regressors use their origin-time values throughout
each fixed-origin forecast. These are conditional persistence scenarios, not known future factors.''',
        'xgboost': '''A separate model learns each horizon's log price change from the origin price.
Lagged gold values, trailing statistics and optional external features are tabular inputs.
Native tree missing-value handling avoids global imputation. Boosting rounds are tuned on inner
forecasting validation; outer blocks are never used for early stopping.''',
        'gru': '''A compact GRU reads a trailing feature window and predicts log changes for all horizons.
The scaler and imputer are fitted within each training fold. Missing future labels are excluded.
A purged chronological tail selects epochs, then a new network is refitted on the full training fold.''',
        'lstm': '''A compact LSTM uses the same data, output targets, folds and initial search budget as GRU.
Only the recurrent cell differs. Each fold starts with fresh weights and fresh fitted preprocessing.
The small initial architecture is deliberate because this dataset contains only a few thousand daily prices.''',
    }[kind]

    cells = [
        markdown(
            f'# Gold forecasting with {title}\n\n{description}\n\n'
            'Run notebook 01 first. Review the shared audit and release '
            'assumptions before interpreting errors.'
        ),
        code(setup),
        markdown('## 1. Load the shared panel and evaluation contract'),
        code(f'''from gold_common_utils import load_prepared, feature_columns, forecast_origins, metrics
from gold_model_utils import (DEFAULT_TRIALS, run_cross_validation, evaluate_holdout,
                              refit_and_forecast, fit_model, predict_model)
KIND = '{kind}'
panel, features, folds = load_prepared()
display(folds)
print('Observed target:', panel.y.notna().sum())'''),
        markdown('''## 2. Inspect input feature groups and candidate settings
`gold_only` establishes whether external factors improve an already useful gold-history model.
The initial search is intentionally small; expand it using development results only.
All candidate selection uses earlier inner validation. Read the adapters in `gold_model_utils.py`
for the explicit training and prediction logic.'''),
        code('''trials = [dict(trial) for trial in DEFAULT_TRIALS[KIND]]
display(pd.DataFrame(trials))
for group in ['gold_only', 'external']:
    print(group, feature_columns(features, group, KIND))
# Change this local list to add candidates; keep comparable GRU/LSTM budgets.'''),
    ]

    # Model-specific third section.
    if kind == 'xgboost':
        cells += [
            markdown('## 3. Check supervised labels and their dates'),
            code('''from gold_model_utils import supervised_rows
cutoff = int(folds.iloc[0].cutoff)
rows, log_changes = supervised_rows(panel, cutoff, cfg['horizons'][-1])
display(pd.DataFrame({'origin': panel.index[rows[-5:]],
                      'target_date': panel.index[rows[-5:] + cfg['horizons'][-1]],
                      'target_log_change': log_changes[-5:]}))
assert (rows + cfg['horizons'][-1] <= cutoff).all()'''),
        ]
    elif kind in ('gru', 'lstm'):
        cells += [
            markdown('## 3. Inspect sequence shapes and architecture'),
            code('''from gold_model_utils import sequence_rows, build_network
example = trials[0]
cutoff = int(folds.iloc[0].cutoff)
rows = sequence_rows(panel, cutoff, example['lookback'], cfg['horizons'])
columns = feature_columns(features, example['group'], KIND)
print('Eligible windows:', len(rows))
print('Input shape: (batch, lookback, features) =', (None, example['lookback'], len(columns)))
print('Output shape:', (None, len(cfg['horizons'])))
network = build_network(KIND, example['lookback'], len(columns), len(cfg['horizons']), example['units'])
network.summary()'''),
        ]
    else:
        cells += [
            markdown('## 3. Inspect observed target and regressor coverage'),
            code('''columns = feature_columns(features, 'external', KIND)
display(features[columns].isna().mean().rename('missing_fraction').to_frame())
panel.y.plot(figsize=(12, 3), title='Observed gold prices; missing labels stay missing')
plt.show()'''),
        ]

    cells += [
        markdown('''## 4. Run nested walk-forward development evaluation
This cell performs actual training and may take time. Each fold reports inner candidate scores,
selects a configuration, refits it, and predicts the outer block. Forecasting uses only information
available at each origin. Outer labels are used only for scoring.

For a quick installation check, use one fold and one candidate. Restore the full settings before
comparing models or making performance claims.'''),
        code('''# Full defaults: five outer folds and all candidates above.
cv_folds = folds
cv_predictions, cv_metrics, selections = run_cross_validation(
    KIND, panel, features, cv_folds, cfg, trials=trials)
display(cv_metrics)
display(selections)'''),
        markdown('## 5. Review errors by horizon, baseline skill, and prediction bias'),
        code('''display(cv_metrics.groupby('horizon')[['MAE', 'RMSE', 'MAPE', 'MASE', 'skill_vs_persistence']].mean())
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
cv_metrics.boxplot(column='MAE', by='horizon', ax=axes[0])
axes[0].set_title('Outer-fold MAE by horizon')
for horizon, group in cv_predictions.groupby('horizon'):
    observed = group.dropna(subset=['actual'])
    axes[1].scatter(observed.actual, observed.prediction - observed.actual, s=12, label=str(horizon))
axes[1].axhline(0, color='black', linewidth=1)
axes[1].set_xlabel('Observed gold price')
axes[1].set_ylabel('Forecast error')
axes[1].legend(title='Horizon')
plt.suptitle('')
plt.tight_layout()'''),
        markdown('''## 6. Plot development forecasts against observed prices
These figures use the fixed-origin predictions from each outer fold:
1. full observed history with every outer prediction;
2. actual vs forecast for each horizon;
3. training context plus test block per fold;
4. actual vs predicted price per horizon.
Each plot is also saved to `artifacts/<model>/plots/`.'''),
        code('''from gold_plot_utils import plot_model_cv
plot_model_cv(panel, cv_predictions, folds, KIND, cfg, ROOT / 'artifacts' / KIND)'''),
        markdown('''## 7. Holdout is intentionally skipped
This run produces future forecasts only. `final_holdout_enabled` stays false, so
no holdout scoring or model ranking is performed. The next cell simply reports
that the holdout was not evaluated.'''),
        code('''cfg = load_config()
if cfg['final_holdout_enabled']:
    holdout_predictions, holdout_metrics, locked_parameters = evaluate_holdout(
        KIND, panel, features, folds, cfg, trials=trials)
    display(holdout_metrics)
    display(locked_parameters)
else:
     print('Holdout not evaluated; this is a forecast-only run.')'''),

        markdown('''## 8. Refit the locked model and generate future forecasts
Run when locked parameters exist. The forecast starts from the latest observed gold quote.
Only the configured horizons are returned. Prophet uses auxiliary future external paths; other models keep origin-time features.
Artifacts include the fitted model, input feature names, preprocessing where applicable,
locked configuration, predictions and metrics. No calibrated uncertainty intervals are claimed.'''),
        code('''locked = ROOT / 'artifacts' / KIND / 'locked_parameters.json'
if locked.exists():
    future_forecast = refit_and_forecast(KIND, panel, features, cfg)
    display(future_forecast)
else:
    print('Final refit waits for the parameter-lock stage.')'''),
        markdown('''## 9. Optional holdout and future forecast plots
These figures appear only when the matching artifacts exist:
- `artifacts/<model>/holdout_predictions.csv` after a locked holdout run;
- `artifacts/<model>/future_forecast.csv` after the final refit.'''),
        code('''from gold_plot_utils import plot_model_holdout, plot_future_forecast
out_dir = ROOT / 'artifacts' / KIND
holdout_path = out_dir / 'holdout_predictions.csv'
future_path = out_dir / 'future_forecast.csv'
if holdout_path.exists():
    holdout_predictions = pd.read_csv(holdout_path, parse_dates=['origin', 'target_date'])
    plot_model_holdout(panel, holdout_predictions, folds, KIND, cfg, out_dir)
else:
    print('No holdout predictions yet.')
if future_path.exists():
    future_forecast = pd.read_csv(future_path, parse_dates=['origin', 'target_date'])
    plot_future_forecast(panel, future_forecast, KIND, cfg, out_dir)
else:
    print('No future forecast yet.')'''),
    ]
    save(f'{number:02d}_gold_{kind}.ipynb', cells)

# ---------------------------------------------------------------------------
# Notebook 06: model comparison on shared observable targets.
# ---------------------------------------------------------------------------
save('06_gold_model_comparison.ipynb', [
markdown('''# Compare gold forecasting models
Run any model notebook first. This notebook reads saved predictions and never retrains models.
Models are compared on the intersection of observed targets and common forecast origins.
The persistence baseline predicts the gold price observed at the origin.'''),
code(setup),
markdown('## 1. Load development results and check coverage'),
code('''from gold_model_utils import compare_results
predictions, results, coverage = compare_results('cv')
display(coverage)
print('Models currently available:', predictions.model.unique().tolist())
display(results)'''),
markdown('## 3. Compare pooled errors and fold stability'),
code('''# Pooled MAE weights individual forecast errors equally within each horizon.
observed = predictions.dropna(subset=['actual', 'prediction']).copy()
observed['absolute_error'] = (observed.prediction - observed.actual).abs()
pooled = observed.groupby(['model', 'horizon']).absolute_error.mean().unstack('horizon')
display(pooled)
pooled.plot.bar(figsize=(12, 4), ylabel='MAE', title='Errors on common observed targets')
plt.tight_layout()
display(results.groupby(['model', 'horizon']).MAE.agg(['mean', 'std', 'min', 'max']))'''),
markdown('## 4. Inspect residuals through time'),
code('''horizon = cfg['horizons'][-1]
fig, ax = plt.subplots(figsize=(13, 4))
for model, group in observed.loc[observed.horizon.eq(horizon)].groupby('model'):
    group.sort_values('target_date').plot(x='target_date', y='absolute_error', ax=ax, label=model)
ax.set_title(f'Absolute error: {horizon}-step forecasts')
plt.tight_layout()'''),
markdown('''## 5. Inspect external-feature selection
Inner scores show which feature group was selected without using the corresponding outer block.
Interpret importance as predictive association, not a causal explanation.'''),
code('''tables = []
for path in sorted((ROOT / 'artifacts').glob('*/inner_selection.csv')):
    table = pd.read_csv(path)
    table['model'] = path.parent.name
    tables.append(table)
if tables:
    selection = pd.concat(tables, ignore_index=True)
    selection['group'] = selection.params.map(lambda value: json.loads(value)['group'])
    display(selection.groupby(['model', 'group']).inner_MAE.agg(['mean', 'std', 'count']))'''),
markdown('''## 6. Review final holdout results if available
Keep these separate from development results. Do not retune using this table.
The current package does not claim uncertainty intervals or verified real-time release vintages.'''),
code('''holdout_files = list((ROOT / 'artifacts').glob('*/holdout_predictions.csv'))
if holdout_files:
    holdout_predictions, holdout_results, holdout_coverage = compare_results('holdout')
    display(holdout_coverage)
    display(holdout_results)
else:
    print('No holdout results yet.')'''),
markdown('''## 7. Optional holdout comparison plots

These figures appear only when all models have observed holdout predictions:
- `holdout_comparison_actual_and_forecasts.png`
- `holdout_comparison_by_fold.png`
- `holdout_comparison_actual_vs_predicted_scatter.png`
- `holdout_comparison_error_ranking.png`

They use the same helper functions as the development comparison plots, with
`stage='holdout'` so only the untouched holdout block is plotted.'''),
code('''holdout_files = sorted((ROOT / 'artifacts').glob('*/holdout_predictions.csv'))
if len(holdout_files) >= 2:
    holdout_predictions = pd.concat(
        [pd.read_csv(path, parse_dates=['origin', 'target_date'])
         for path in holdout_files], ignore_index=True)
    out_dir = ROOT / 'artifacts' / 'comparison'
    plot_comparison_actual_and_forecasts(holdout_predictions, panel, folds, cfg, out_dir, stage='holdout')
    plot_comparison_by_fold(holdout_predictions, panel, folds, cfg, out_dir, stage='holdout')
    plot_comparison_scatter(holdout_predictions, panel, folds, cfg, out_dir, stage='holdout')
    plot_comparison_error_ranking(holdout_predictions, cfg, out_dir, stage='holdout')
else:
    print('Need at least two models with holdout predictions to compare them.')'''),
])
print('Created six notebooks.')
