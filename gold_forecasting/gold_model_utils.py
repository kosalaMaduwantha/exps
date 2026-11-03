"""Readable model adapters and a shared nested walk-forward evaluation loop.

Design rules
------------
- Optional heavy dependencies (TensorFlow, Prophet, XGBoost) are imported only
  by the adapter that needs them, so preprocessing and comparison stay usable
  without every backend installed.
- Every fold fits fresh preprocessing and a fresh model. Nothing learned on
  outer evaluation data is ever reused for selection or refitting.
- Fixed-origin forecasts use external values known at the origin. The final
  Prophet forecast may use auxiliary future external paths; future market/gold
  realized values are never read.
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gold_common_utils import (
    ROOT,
    feature_columns,
    forecast_origins,
    metrics,
    predictions_frame,
    seed_everything,
)

# Initial candidate search per model. Deliberately small; expand only after
# reviewing development folds and keep GRU/LSTM budgets comparable.
DEFAULT_TRIALS = {
    'prophet': [
        {'group': 'gold_only', 'changepoint_prior_scale': 0.05},
        {'group': 'external', 'changepoint_prior_scale': 0.05},
        {'group': 'external', 'changepoint_prior_scale': 0.2},
    ],
    'xgboost': [
        {'group': 'gold_only', 'max_depth': 2, 'n_estimators': 250},
        {'group': 'external', 'max_depth': 2, 'n_estimators': 250},
        {'group': 'external', 'max_depth': 3, 'n_estimators': 400},
    ],
    'gru': [
        {'group': 'gold_only', 'lookback': 20, 'units': 32},
        {'group': 'external', 'lookback': 20, 'units': 32},
        {'group': 'external', 'lookback': 60, 'units': 32},
    ],
    'lstm': [
        {'group': 'gold_only', 'lookback': 20, 'units': 32},
        {'group': 'external', 'lookback': 20, 'units': 32},
        {'group': 'external', 'lookback': 60, 'units': 32},
    ],
}


def supervised_rows(panel, cutoff, horizon):
    """Return eligible training origins and their log-change labels.

    A training row is eligible only when both the origin and its target are
    observed, and the target date does not exceed the training cutoff. Labels
    are ``log(price_after_h) / log(price_now)`` shifts, matching how the model
    predicts relative moves rather than raw levels.
    """
    origins = np.arange(60, cutoff - horizon + 1)
    valid = (
        panel.y.iloc[origins].notna().to_numpy()
        & panel.y.iloc[origins + horizon].notna().to_numpy()
    )
    origins = origins[valid]
    target = np.log(
        panel.y.iloc[origins + horizon].to_numpy()
        / panel.y.iloc[origins].to_numpy()
    )
    return origins, target


def fit_xgboost(panel, x, cutoff, params, cfg):
    """Train one tree model per horizon on rows inside the training cutoff."""
    from xgboost import XGBRegressor

    columns = feature_columns(x, params['group'], 'xgboost')
    models = []
    for horizon in cfg['horizons']:
        rows, target = supervised_rows(panel, cutoff, horizon)
        if len(rows) < 100:
            raise ValueError('Fewer than 100 eligible training examples.')
        model = XGBRegressor(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,
            reg_lambda=5,
            objective='reg:squarederror',
            tree_method='hist',
            n_jobs=2,
            random_state=cfg['seed'],
        )
        # Native missing-value handling: no fitted scaler is needed. Boosting
        # rounds are fixed by the inner validation search, not outer stopping.
        model.fit(x.iloc[rows][columns], target)
        models.append(model)
    return {
        'kind': 'xgboost',
        'models': models,
        'columns': columns,
        'params': params,
    }


def fit_prophet(panel, x, cutoff, params, cfg):
    """Fit Prophet on log gold prices with optional external regressors."""
    from prophet import Prophet

    columns = feature_columns(x, params['group'], 'prophet')
    train = pd.DataFrame(
        {
            'ds': panel.index[:cutoff+1],
            'y': panel.y.iloc[:cutoff+1].to_numpy(),
        }
    )
    # Log scale preserves positivity and lets trend changes act proportionally.
    train['y'] = np.log(train.y)
    medians = x.iloc[:cutoff+1][columns].median().fillna(0)
    for column in columns:
        train[column] = (
            x.iloc[:cutoff+1][column].fillna(medians[column]).to_numpy()
        )
    model = Prophet(
        changepoint_prior_scale=params['changepoint_prior_scale'],
        seasonality_prior_scale=1.0,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        uncertainty_samples=0,
    )
    kept_columns = []
    for column in columns:
        # Constant regressors are unidentified inside a fold and can be skipped.
        if train.loc[train.y.notna(), column].nunique() > 1:
            model.add_regressor(column, prior_scale=0.5)
            kept_columns.append(column)
    model.fit(train[['ds', 'y'] + kept_columns].dropna(subset=['y']))
    return {
        'kind': 'prophet',
        'model': model,
        'columns': kept_columns,
        'medians': medians,
        'params': params,
    }


def sequence_rows(panel, cutoff, lookback, horizons):
    """Return eligible recurrent-training origins and their lookback windows.

    An origin is eligible only when the whole trailing window and every target
    horizon are observed. Long gaps are never bridged with fill or imputation,
    so a network never trains on an imputed gold history.
    """
    origins = np.arange(max(60, lookback - 1), cutoff - max(horizons) + 1)
    valid = panel.y.iloc[origins].notna().to_numpy(copy=True)
    for horizon in horizons:
        valid &= panel.y.iloc[origins + horizon].notna().to_numpy()

    # Reject any window that crosses at least a `lookback`-length target gap.
    long_gap = panel.y.isna().rolling(6, min_periods=6).sum().ge(6)
    for i, origin in enumerate(origins):
        valid[i] &= not long_gap.iloc[origin-lookback+1:origin+1].any()
    return origins[valid]


def windows(array, origins, lookback):
    """Stack trailing lookback windows for recurrent inputs."""
    return np.stack(
        [array[o-lookback+1:o+1] for o in origins]
    ).astype('float32')


def build_network(kind, lookback, n_features, n_outputs, units):
    """Assemble the small shared GRU/LSTM architecture."""
    import tensorflow as tf

    recurrent = tf.keras.layers.GRU if kind == 'gru' else tf.keras.layers.LSTM
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(lookback, n_features)),
            recurrent(units, dropout=0.1),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(n_outputs),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
        loss='mse',
    )
    return model


def fit_recurrent(kind, panel, x, cutoff, params, cfg):
    """Fit one recurrent model on this fold and refit fresh at the best epoch."""
    import tensorflow as tf

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(cfg['seed'])
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass

    columns = feature_columns(x, params['group'], kind)
    lookback = params['lookback']
    origins = sequence_rows(panel, cutoff, lookback, cfg['horizons'])
    if len(origins) < 100:
        raise ValueError('Insufficient complete target windows for recurrent training.')

    # Epoch selection uses a chronological tail inside this training fold only;
    # training origins are purged so their targets cannot reach that tail.
    split_origin = origins[int(len(origins) * 0.85)]
    train_origins = origins[origins + max(cfg['horizons']) < split_origin]
    val_origins = origins[origins >= split_origin]

    preprocessor = make_pipeline(
        SimpleImputer(strategy='median', keep_empty_features=True),
        StandardScaler(),
    )
    preprocessor.fit(x.iloc[:split_origin][columns])
    transformed = preprocessor.transform(x.iloc[:cutoff+1][columns])
    anchor = panel.y.to_numpy()

    def labels(rows):
        """Log changes from each origin to every configured horizon."""
        return np.column_stack(
            [np.log(anchor[rows+h] / anchor[rows]) for h in cfg['horizons']]
        )

    target_scaler = StandardScaler().fit(labels(train_origins))
    model = build_network(
        kind, lookback, transformed.shape[1], len(cfg['horizons']), params['units']
    )
    # Explicit batches avoid silent validation_slice boundary issues and keep
    # the chronological split observable.
    train_x = windows(transformed, train_origins, lookback)
    val_x = windows(transformed, val_origins, lookback)
    train_y = target_scaler.transform(labels(train_origins)).astype('float32')
    val_y = target_scaler.transform(labels(val_origins)).astype('float32')
    history = model.fit(
        train_x,
        train_y,
        validation_data=(val_x, val_y),
        epochs=cfg['max_epochs'],
        batch_size=32,
        shuffle=False,
        verbose=0,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=cfg['patience'],
                restore_best_weights=True,
            )
        ],
    )
    best_epochs = int(np.argmin(history.history['val_loss']) + 1)

    # Reinitialize and refit on all eligible rows with the chosen epoch count.
    # No outer-validation labels or scaler statistics are ever reused here.
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(cfg['seed'])
    preprocessor.fit(x.iloc[:cutoff+1][columns])
    transformed = preprocessor.transform(x.iloc[:cutoff+1][columns])
    target_scaler.fit(labels(origins))
    model = build_network(
        kind, lookback, transformed.shape[1], len(cfg['horizons']), params['units']
    )
    model.fit(
        windows(transformed, origins, lookback),
        target_scaler.transform(labels(origins)).astype('float32'),
        epochs=best_epochs,
        batch_size=32,
        shuffle=False,
        verbose=0,
    )
    return {
        'kind': kind,
        'model': model,
        'columns': columns,
        'preprocessor': preprocessor,
        'target_scaler': target_scaler,
        'params': params,
        'best_epochs': best_epochs,
        'history': history.history,
    }


def fit_model(kind, panel, x, cutoff, params, cfg):
    """Dispatch to one model adapter and attach fold-level metadata."""
    seed_everything(cfg['seed'])
    if kind == 'xgboost':
        bundle = fit_xgboost(panel, x, cutoff, params, cfg)
    elif kind == 'prophet':
        bundle = fit_prophet(panel, x, cutoff, params, cfg)
    elif kind in ('gru', 'lstm'):
        bundle = fit_recurrent(kind, panel, x, cutoff, params, cfg)
    else:
        raise ValueError(f'Unknown model kind: {kind}')
    bundle['training_config'] = dict(cfg)
    bundle['training_cutoff'] = str(panel.index[cutoff].date())
    return bundle


def forecast_external_paths(panel, x, origin, columns, cfg):
    """Forecast anchored future external regressor values with Prophet.

    Each regressor uses its own daily as-of history through ``origin``. The
    fitted coefficients of the main gold model stay frozen while the future
    macro inputs follow their own forecast paths. Paths are anchored so the
    first forecast step continues from the last observed regressor value.
    """
    from prophet import Prophet

    future_dates = pd.bdate_range(
        panel.index[origin] + pd.offsets.BDay(1),
        periods=max(cfg['horizons']),
    )
    paths = pd.DataFrame({'ds': future_dates})
    for column in columns:
        history = x[column].iloc[:origin + 1].dropna()
        if history.empty:
            paths[column] = np.nan
            continue
        if len(history) < 30 or history.nunique() <= 1:
            paths[column] = history.iloc[-1]
            continue
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            uncertainty_samples=0,
            changepoint_prior_scale=0.05,
        )
        model.fit(pd.DataFrame({'ds': history.index, 'y': history.to_numpy()}))
        predict_dates = pd.DatetimeIndex([history.index[-1]]).append(future_dates)
        forecast = model.predict(pd.DataFrame({'ds': predict_dates}))
        adjustment = history.iloc[-1] - forecast['yhat'].iloc[0]
        paths[column] = forecast['yhat'].iloc[1:].to_numpy() + adjustment
    return paths


def predict_model(bundle, panel, x, origins, cfg, future_regressors=None):
    """Produce fixed-origin forecasts for every configured horizon."""
    columns = bundle['columns']
    if bundle['training_config']['horizons'] != cfg['horizons']:
        raise ValueError('Prediction horizons differ from the trained model.')
    kind = bundle['kind']
    anchor = panel.y.iloc[origins].to_numpy()

    if future_regressors is not None and len(origins) != 1:
        raise ValueError('Future regressor paths require exactly one origin.')

    if kind == 'xgboost':
        returns = np.column_stack(
            [model.predict(x.iloc[origins][columns]) for model in bundle['models']]
        )
        return anchor[:, None] * np.exp(returns)

    if kind in ('gru', 'lstm'):
        transformed = bundle['preprocessor'].transform(x[columns])
        inputs = windows(transformed, origins, bundle['params']['lookback'])
        # Calling the model directly avoids an unnecessary data-loader thread
        # pool and keeps prediction deterministic.
        standardized = bundle['model'](inputs, training=False).numpy()
        returns = bundle['target_scaler'].inverse_transform(standardized)
        return anchor[:, None] * np.exp(returns)

    paths = None
    if future_regressors is not None:
        paths = future_regressors.copy()
        if 'ds' in paths.columns:
            paths = paths.set_index('ds')
        paths.index = pd.to_datetime(paths.index)

    predictions = []
    for origin in origins:
        dates = pd.bdate_range(
            panel.index[origin] + pd.offsets.BDay(1), periods=max(cfg['horizons'])
        )
        future = pd.DataFrame({'ds': dates})
        for column in columns:
            value = x.iloc[origin][column]
            fallback = value if pd.notna(value) else bundle['medians'][column]
            if paths is not None and column in paths.columns:
                series = paths[column].reindex(dates)
                future[column] = series.where(series.notna(), fallback).to_numpy()
            else:
                future[column] = fallback
        forecast = bundle['model'].predict(future)
        predictions.append(
            [np.exp(forecast.yhat.iloc[h-1]) for h in cfg['horizons']]
        )
    return np.asarray(predictions)


def score_predictions(predictions):
    """Report a single comparable score per fold on observed labels only."""
    table = metrics(predictions)
    if len(table) == 0:
        raise ValueError('No observed targets available for scoring.')
    return table.groupby('horizon').MAE.mean().mean()


def select_parameters(kind, panel, x, cutoff, cfg, trials=None):
    """Tune the model configuration on an earlier inner block only.

    Reviewers must see the inner scores before the outer block is ever used.
    This keeps the outer data reserved strictly for reporting.
    """
    inner_cutoff = cutoff - cfg['validation_steps']
    origins = forecast_origins(panel, inner_cutoff, cutoff, cfg)
    if len(origins) == 0:
        raise ValueError('No eligible inner-validation origins.')

    records = []
    candidates = trials or DEFAULT_TRIALS[kind]
    for candidate_id, params in enumerate(candidates):
        bundle = fit_model(kind, panel, x, inner_cutoff, params, cfg)
        values = predict_model(bundle, panel, x, origins, cfg)
        predictions = predictions_frame(
            panel, origins, values, kind, 'inner', inner_cutoff, cfg
        )
        score = score_predictions(predictions)
        records.append(
            {
                'candidate': candidate_id,
                'inner_MAE': score,
                'params': json.dumps(params),
            }
        )
        print(
            f'  candidate {candidate_id + 1}/{len(candidates)}: '
            f'inner MAE={score:.3f}',
            flush=True,
        )

    table = pd.DataFrame(records)
    best = int(table.loc[table.inner_MAE.idxmin(), 'candidate'])
    return dict(candidates[best]), table


def run_cross_validation(kind, panel, x, folds, cfg, trials=None):
    """Run nested walk-forward development evaluation for one model kind.

    For each non-holdout fold:
    1. tune parameters on an earlier inner block,
    2. refit on the full training cutoff with the selected parameters,
    3. forecast the outer block from fixed origins,
    4. checkpoint predictions/selection/timing to disk after every fold.
    """
    output = ROOT / 'artifacts' / kind
    output.mkdir(parents=True, exist_ok=True)
    predictions, selections, timing = [], [], []

    for fold in folds.loc[folds.fold.ne('holdout')].itertuples(index=False):
        started = time.perf_counter()
        print(
            f'{kind}: {fold.fold}; train through {panel.index[fold.cutoff].date()}',
            flush=True,
        )
        params, selection = select_parameters(kind, panel, x, fold.cutoff, cfg, trials)
        selection['fold'] = fold.fold
        selections.append(selection)

        bundle = fit_model(kind, panel, x, fold.cutoff, params, cfg)
        origins = forecast_origins(panel, fold.cutoff, fold.stop, cfg)
        values = predict_model(bundle, panel, x, origins, cfg)
        predictions.append(
            predictions_frame(
                panel, origins, values, kind, fold.fold, fold.cutoff, cfg
            )
        )

        timing.append(
            {
                'fold': fold.fold,
                'seconds_including_selection': time.perf_counter() - started,
                'parameters': json.dumps(params),
                'network_parameters': (
                    bundle['model'].count_params()
                    if kind in ('gru', 'lstm')
                    else None
                ),
            }
        )
        pd.DataFrame(timing).to_csv(output / 'training_log.csv', index=False)
        # Checkpointing after each fold means an interrupted run keeps a
        # complete, human-reviewable subset of results.
        pd.concat(predictions).to_csv(output / 'cv_predictions.csv', index=False)
        pd.concat(selections).to_csv(output / 'inner_selection.csv', index=False)

    predictions = pd.concat(predictions, ignore_index=True)
    result = metrics(predictions)
    result.to_csv(output / 'cv_metrics.csv', index=False)
    return predictions, result, pd.concat(selections, ignore_index=True)


def evaluate_holdout(kind, panel, x, folds, cfg, trials=None):
    """Evaluate the untouched final fold once, after locking parameters."""
    if not cfg['final_holdout_enabled']:
        raise ValueError(
            'Review development results, then set final_holdout_enabled=true '
            'in config.json.'
        )
    output = ROOT / 'artifacts' / kind
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'holdout_predictions.csv').exists():
        raise FileExistsError(
            'Holdout already evaluated. Existing results are preserved; '
            'do not repeatedly tune on them.'
        )

    fold = folds.loc[folds.fold.eq('holdout')].iloc[0]
    cutoff, stop = int(fold.cutoff), int(fold.stop)
    params, selection = select_parameters(kind, panel, x, cutoff, cfg, trials)

    # Persist the selected parameters before any holdout score is revealed.
    (output / 'locked_parameters.json').write_text(json.dumps(params, indent=2))
    selection.to_csv(output / 'final_inner_selection.csv', index=False)

    bundle = fit_model(kind, panel, x, cutoff, params, cfg)
    origins = forecast_origins(panel, cutoff, stop, cfg)
    values = predict_model(bundle, panel, x, origins, cfg)
    predictions = predictions_frame(
        panel, origins, values, kind, 'holdout', cutoff, cfg
    )
    predictions.to_csv(output / 'holdout_predictions.csv', index=False)
    metrics(predictions).to_csv(output / 'holdout_metrics.csv', index=False)
    save_bundle(bundle, output / 'holdout_model')
    return predictions, metrics(predictions), params


def save_bundle(bundle, directory):
    """Persist one trained model and everything needed to reload it."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {
        'kind': bundle['kind'],
        'columns': bundle['columns'],
        'params': bundle['params'],
        'training_config': bundle['training_config'],
        'training_cutoff': bundle['training_cutoff'],
    }
    (directory / 'metadata.json').write_text(json.dumps(metadata, indent=2))

    if bundle['kind'] == 'prophet':
        from prophet.serialize import model_to_json

        (directory / 'prophet.json').write_text(model_to_json(bundle['model']))
        joblib.dump(bundle['medians'], directory / 'medians.joblib')
    elif bundle['kind'] == 'xgboost':
        for horizon_id, model in enumerate(bundle['models']):
            model.save_model(directory / f'horizon_{horizon_id}.json')
    else:
        bundle['model'].save(directory / 'network.keras')
        joblib.dump(
            {
                'preprocessor': bundle['preprocessor'],
                'target_scaler': bundle['target_scaler'],
            },
            directory / 'preprocessing.joblib',
        )
        (directory / 'learning_history.json').write_text(
            json.dumps(bundle['history'])
        )


def load_bundle(directory):
    """Load a model bundle saved by ``save_bundle``."""
    directory = Path(directory)
    bundle = json.loads((directory / 'metadata.json').read_text())

    if bundle['kind'] == 'prophet':
        from prophet.serialize import model_from_json

        bundle['model'] = model_from_json(
            (directory / 'prophet.json').read_text()
        )
        bundle['medians'] = joblib.load(directory / 'medians.joblib')
    elif bundle['kind'] == 'xgboost':
        from xgboost import XGBRegressor

        files = sorted(
            directory.glob('horizon_*.json'),
            key=lambda path: int(path.stem.split('_')[-1]),
        )
        bundle['models'] = []
        for path in files:
            model = XGBRegressor()
            model.load_model(path)
            bundle['models'].append(model)
    else:
        import tensorflow as tf

        bundle['model'] = tf.keras.models.load_model(
            directory / 'network.keras'
        )
        bundle.update(joblib.load(directory / 'preprocessing.joblib'))
    return bundle


def refit_and_forecast(kind, panel, x, cfg):
    """Refit the locked configuration and forecast future horizons once.

    The forecast starts from the latest observed gold quote. Prophet uses
    auxiliary daily as-of paths for its external regressors; other models keep
    origin-time features. Generates no calibrated uncertainty interval.
    """
    output = ROOT / 'artifacts' / kind
    params_path = output / 'locked_parameters.json'
    if not params_path.exists():
        raise FileNotFoundError(
            'Run the final parameter lock and holdout stage first.'
        )
    params = json.loads(params_path.read_text())
    cutoff = int(np.flatnonzero(panel.y.notna().to_numpy())[-1])
    bundle = fit_model(kind, panel, x, cutoff, params, cfg)

    future_regressors = None
    if kind == 'prophet':
        future_regressors = forecast_external_paths(
            panel, x, cutoff, bundle['columns'], cfg
        )
        future_regressors.to_csv(output / 'future_regressors.csv', index=False)

    values = predict_model(
        bundle, panel, x, np.array([cutoff]), cfg,
        future_regressors=future_regressors,
    )[0]
    save_bundle(bundle, output / 'final_model')

    origin_price = float(panel.y.iloc[cutoff])
    forecast = pd.DataFrame(
        {
            'origin': panel.index[cutoff],
            'horizon': cfg['horizons'],
            'horizon_label': cfg['horizon_labels'],
            'target_date': [
                panel.index[cutoff] + pd.offsets.BDay(h) for h in cfg['horizons']
            ],
            'origin_price': origin_price,
            'forecast': values,
            'change_from_origin_pct': (values / origin_price - 1) * 100,
            'model': kind,
        }
    )
    forecast.to_csv(output / 'future_forecast.csv', index=False)
    return forecast


def compare_results(stage='cv'):
    """Compare saved model forecasts on a common set of eligible targets.

    Only origins/targets present in every model's table are pooled together,
    so rankings do not depend on models having unequal missing-label coverage.
    Persistence and trailing-drift baselines are appended to the comparison.
    """
    files = sorted((ROOT / 'artifacts').glob(f'*/{stage}_predictions.csv'))
    if not files:
        raise FileNotFoundError(
            'No prediction files yet. Run model notebooks first.'
        )

    frames = [
        pd.read_csv(path, parse_dates=['origin', 'target_date'])
        for path in files
    ]
    all_predictions = pd.concat(frames, ignore_index=True)

    keys = ['fold', 'origin', 'target_date', 'horizon']
    eligible = [
        frame.dropna(subset=['actual', 'prediction']).set_index(keys).index
        for frame in frames
    ]
    common = eligible[0]
    for index in eligible[1:]:
        common = common.intersection(index)
    comparison = (
        all_predictions.set_index(keys)
        .loc[lambda d: d.index.isin(common)]
        .reset_index()
    )
    if comparison.empty:
        raise ValueError('No common eligible predictions across models.')

    # Append the deterministic persistence and drift baselines for context.
    baseline = comparison.drop_duplicates(keys).copy()
    baseline['prediction'] = baseline.baseline
    baseline['model'] = 'persistence'
    drift = comparison.drop_duplicates(keys).copy()
    drift['prediction'] = drift['drift']
    drift['model'] = 'trailing_drift'
    comparison = pd.concat([comparison, baseline, drift], ignore_index=True)

    summary = metrics(comparison)
    coverage = all_predictions.groupby('model').agg(
        total_predictions=('prediction', 'size'),
        observed_targets=('actual', 'count'),
    )
    coverage['common_predictions'] = len(common)

    destination = ROOT / 'artifacts/comparison'
    destination.mkdir(exist_ok=True)
    summary.to_csv(destination / f'{stage}_metrics_common_targets.csv', index=False)
    coverage.to_csv(destination / f'{stage}_coverage.csv')
    return comparison, summary, coverage
