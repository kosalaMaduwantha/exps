"""Shared, auditable preprocessing and fixed-origin evaluation for gold forecasts.

Scope and guarantees
--------------------
- Every feature row is indexed by forecast origin ``t`` and every label refers
  to ``t + h``. Nothing from after an origin may enter its features.
- No fitted imputer, scaler, or model is shared between validation folds. Each
  fold refits its own preprocessing before scoring.
- Raw files in ``data/raw`` are read-only. The ``preprocess`` helper writes
  standardized tables to ``data/processed`` for the model notebooks.

The module is intentionally dependency-light: it imports pandas/numpy only, so
notebooks that do not need TensorFlow, Prophet, or XGBoost can still run.
"""
import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Project directory containing config.json, data/, notebooks, and artifacts.
ROOT = Path(__file__).resolve().parent


def load_config():
    """Read the shared experiment configuration from config.json."""
    cfg = json.loads((ROOT / 'config.json').read_text())
    if len(cfg.get('horizon_labels', [])) != len(cfg['horizons']):
        raise ValueError('horizon_labels and horizons must have the same length.')
    return cfg


def numeric(values):
    """Coerce comma/percent-formatted source text into numeric values."""
    cleaned = (
        values.astype(str)
        .str.replace(',', '', regex=False)
        .str.replace('%', '', regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors='coerce')


def parse_gold_date(value):
    """Parse known gold-file formats explicitly; never guess malformed dates.

    Supported inputs are pandas/numpy datetimes, Excel serial numbers,
    ``DD.MM.YYYY``, ``DD/MM/YYYY``, ISO 8601, and ``YYYYMMDD``. Values that do
    not match any known format return ``NaT`` so the cleaning audit can
    quarantine them for manual review.
    """
    if pd.isna(value):
        return pd.NaT
    # Datetime-like objects (including Excel readers) are used directly.
    if isinstance(value, (pd.Timestamp, np.datetime64)) or hasattr(value, 'year'):
        return pd.Timestamp(value).normalize()
    # Integers/floats are Excel serial dates, not all-purpose day offsets.
    if isinstance(value, (int, float)):
        if 20000 <= value <= 60000:
            return pd.Timestamp('1899-12-30') + pd.Timedelta(days=float(value))
        return pd.NaT
    # Strings: normalize whitespace, non-breaking spaces, and repeated dots.
    value = str(value).strip().replace('\xa0', '')
    value = re.sub(r'\.{2,}', '.', value)
    for fmt in ('%d.%m.%Y', '%d/%m/%Y', '%Y-%m-%d', '%Y%m%d'):
        parsed = pd.to_datetime(value, format=fmt, errors='coerce')
        if pd.notna(parsed):
            return parsed.normalize()
    return pd.NaT


def clean_gold(raw_dir, cfg):
    """Clean the gold workbook and return the target plus a row-level audit.

    Date reconciliation rules
    -------------------------
    - If both date columns are valid and agree, the later column wins.
    - If the columns disagree, the row is treated as unresolved.
    - If only one column supplies an in-range date, that value is used and the
      resolution is recorded in ``date_resolution``.

    Target rules
    ------------
    - Weekend rows and out-of-bounds prices are quarantined, not deleted.
    - Conflicting duplicate prices for one date keep none of the copies.
    - Exact duplicates keep the first source row.
    """
    raw = pd.read_excel(raw_dir / 'gold_data_new.xlsx', header=None).iloc[3:].copy()
    # One-based Excel row number helps humans trace every retained/dropped row.
    raw['source_row'] = raw.index + 1
    raw['date_a'] = raw[0].map(parse_gold_date)
    raw['date_b'] = raw[1].map(parse_gold_date)

    start = pd.Timestamp(cfg['date_start'])
    end = pd.Timestamp(cfg['date_end'])
    good_a = raw.date_a.between(start, end)
    good_b = raw.date_b.between(start, end)

    # Prefer the second date column; recover from the first only when needed.
    raw['ds'] = raw.date_b.where(good_b, raw.date_a.where(good_a))
    mismatch = good_a & good_b & raw.date_a.ne(raw.date_b)
    raw.loc[mismatch, 'ds'] = pd.NaT
    raw['date_resolution'] = 'second_date_column'
    raw.loc[~good_b & good_a, 'date_resolution'] = 'recovered_from_first_date_column'
    raw.loc[raw.ds.isna(), 'date_resolution'] = 'unresolved'

    # Target values: reject obvious invalid prices and weekend rows.
    raw['y'] = numeric(raw[cfg['target_column']])
    low, high = cfg['target_bounds']
    raw['reason'] = ''
    raw.loc[raw.ds.isna(), 'reason'] = 'unresolved_date'
    raw.loc[
        raw.ds.notna() & ~raw.y.between(low, high), 'reason'
    ] = 'missing_or_out_of_bounds_target'
    raw.loc[
        raw.ds.notna() & (raw.ds.dt.dayofweek >= 5), 'reason'
    ] = 'weekend_requires_review'

    # Quarantine conflicting duplicate prices; keep the first exact duplicate.
    eligible = raw.reason.eq('')
    distinct = raw.loc[eligible].groupby('ds').y.nunique()
    conflicts = distinct.index[distinct > 1]
    raw.loc[eligible & raw.ds.isin(conflicts), 'reason'] = 'conflicting_duplicate_requires_review'
    eligible = raw.reason.eq('')
    duplicate = raw.loc[eligible].duplicated('ds', keep='first')
    raw.loc[duplicate.index[duplicate], 'reason'] = 'exact_target_duplicate'

    clean = raw.loc[raw.reason.eq(''), ['ds', 'y']].sort_values('ds').set_index('ds')

    # Robust trailing diagnostics flag possible errors, but do not delete real
    # shocks. The window uses only past rows (``shift(1)``) before the flag.
    changes = clean.y.diff()
    median = changes.rolling(60, min_periods=20).median().shift(1)
    mad = changes.rolling(60, min_periods=20).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True
    ).shift(1)
    clean['large_change_flag'] = (
        (changes - median).abs() > 8 * 1.4826 * mad.replace(0, np.nan)
    )

    audit = raw[
        ['source_row', 0, 1, 'date_a', 'date_b', 'ds', 'y', 'reason', 'date_resolution']
    ].copy()
    audit.columns = [
        'source_row',
        'raw_date_a',
        'raw_date_b',
        'date_a',
        'date_b',
        'ds',
        'y',
        'reason',
        'date_resolution',
    ]
    return clean, audit


def available_series(values, dates, available_dates, calendar, max_age_days):
    """Backward as-of join of one release series onto the weekday calendar.

    Returns the carried values and the age (calendar days) between each
    calendar row and the underlying observation date. Availability dates may
    fall on weekends, so the join uses the last release up to each origin.
    Values vanish once they exceed ``max_age_days`` so stale releases cannot
    silently substitute for fresh data.
    """
    releases = pd.DataFrame(
        {
            'observation_date': pd.to_datetime(dates),
            'available_date': pd.to_datetime(available_dates),
            'value': np.asarray(values, dtype=float),
        }
    ).dropna()
    releases = releases.sort_values('available_date').drop_duplicates(
        'available_date', keep='last'
    )
    joined = pd.merge_asof(
        pd.DataFrame({'ds': calendar}),
        releases,
        left_on='ds',
        right_on='available_date',
        direction='backward',
    )
    age = (joined.ds - joined.observation_date).dt.days
    value = joined.value.where(age <= max_age_days)
    return (
        pd.Series(value.to_numpy(), index=calendar),
        pd.Series(age.to_numpy(), index=calendar),
    )


def build_panel(raw_dir, cfg):
    """Build one weekday panel containing the target and external inputs.

    Every external series is pulled through an assumed availability date.
    The availability dates are documented assumptions because historical
    release vintages are unavailable. They preserve causality for the
    snapshot-based backtests.
    """
    gold, audit = clean_gold(raw_dir, cfg)

    # Weekday calendar; horizons count these slots, not exchange sessions.
    calendar = pd.bdate_range(cfg['date_start'], cfg['date_end'], name='ds')
    panel = gold.reindex(calendar)
    panel['target_observed'] = panel.y.notna().astype(int)
    inventory = []

    def add(name, values, dates, frequency='daily', lag=1, max_age=10):
        """Attach one series plus its age column and record its assumptions."""
        dates = pd.DatetimeIndex(pd.to_datetime(dates))
        if frequency == 'monthly':
            # Month-end observation becomes available after the lag days.
            availability = dates.to_period('M').to_timestamp('M') + pd.Timedelta(days=lag)
        elif frequency == 'daily':
            availability = dates + pd.offsets.BDay(lag)
        else:
            availability = dates + pd.Timedelta(days=lag)
        panel[name], panel[name + '_age_days'] = available_series(
            values, dates, availability, calendar, max_age
        )
        inventory.append(
            {
                'series': name,
                'frequency': frequency,
                'lag': lag,
                'staleness_limit_calendar_days': max_age,
                'availability_status': 'assumed; historical vintages unavailable',
            }
        )

    # Daily market closes: Brent, WTI, and the 10-year yield.
    for name, filename in [
        ('brent', 'brent_historical.csv'),
        ('wti', 'wti_historical.csv'),
    ]:
        frame = pd.read_csv(raw_dir / filename)
        dates = pd.to_datetime(frame.Date, format='%A, %B %d, %Y')
        add(
            name,
            numeric(frame.Close),
            dates,
            lag=cfg['daily_market_lag_business_days'],
        )

    # Backup Brent instrument: the EIA spot series, kept for comparison.
    frame = pd.read_csv(raw_dir / 'Europe_Brent_Spot_Price_FOB.csv', skiprows=4)
    add(
        'brent_spot',
        numeric(frame.iloc[:, 1]),
        pd.to_datetime(frame.iloc[:, 0], format='%m/%d/%Y'),
        lag=cfg['daily_market_lag_business_days'],
    )

    # US 10-year yield.
    frame = pd.read_csv(raw_dir / 'United States 10-Year Bond Yield Historical Data-use.csv')
    add(
        'yield10',
        numeric(frame.Price),
        pd.to_datetime(frame.Date, format='%m/%d/%Y'),
        lag=cfg['daily_market_lag_business_days'],
    )

    # US Dollar Index. The final monthly candle could be incomplete (month not
    # finished at the configured END), so it is dropped. its own lag is one
    # month-end plus one calendar day.
    frame = pd.read_csv(raw_dir / 'US Dollar Index Historical Data-use.csv')
    dates = pd.to_datetime(frame.Date, format='%m/%d/%Y')
    complete = dates.dt.to_period('M') < pd.Timestamp(cfg['date_end']).to_period('M')
    add(
        'dxy',
        numeric(frame.loc[complete, 'Price']),
        dates[complete],
        'monthly',
        lag=1,
        max_age=65,
    )

    # Fed funds: the CSV source and its near-duplicate are cross-checked, and
    # only one copy is used as a feature.
    frame = pd.read_csv(raw_dir / 'FEDFUNDS.csv')
    duplicate = pd.read_csv(raw_dir / 'US-interest-rates-use.csv')
    assert np.array_equal(frame.iloc[:, 0], duplicate.iloc[:, 0])
    assert np.allclose(frame.iloc[:, 1], duplicate.iloc[:, 1])
    add(
        'fed',
        numeric(frame.FEDFUNDS),
        pd.to_datetime(frame.observation_date),
        'monthly',
        cfg['fed_release_lag_calendar_days'],
        75,
    )

    # CPI: reindex to a complete monthly calendar before computing changes so
    # 12 steps always span twelve reference months even with a missing row.
    frame = pd.read_excel(raw_dir / 'SeriesReport-20261028041515_7e01eb-use.xlsx', header=11)
    frame = frame.loc[frame.Period.astype(str).str.fullmatch(r'M(0[1-9]|1[0-2])')].copy()
    dates = pd.to_datetime(frame.Year.astype(int).astype(str) + '-' + frame.Period.str[1:] + '-01')
    cpi = pd.Series(numeric(frame.Value).to_numpy(), index=dates).sort_index()
    cpi = cpi.reindex(pd.date_range(cpi.index.min(), cpi.index.max(), freq='MS'))
    for name, values in [
        ('cpi_yoy', cpi.pct_change(12, fill_method=None) * 100),
        ('cpi_mom', cpi.pct_change(fill_method=None) * 100),
    ]:
        add(
            name,
            values,
            values.index,
            'monthly',
            cfg['cpi_release_lag_calendar_days'],
            80,
        )

    # GPR: daily levels and 7-day trailing means; both are released with a lag.
    frame = pd.read_excel(raw_dir / 'data_gpr_daily_recent.xls')
    dates = pd.to_datetime(frame.Date)
    for column, name in [
        ('GPRD', 'gpr'),
        ('GPRD_ACT', 'gpr_acts'),
        ('GPRD_THREAT', 'gpr_threats'),
    ]:
        values = numeric(frame[column])
        add(
            name,
            values,
            dates,
            'calendar_daily',
            cfg['gpr_release_lag_calendar_days'],
            20,
        )
        trailing = pd.Series(values.to_numpy(), index=dates).sort_index().rolling(
            '7D', min_periods=3
        ).mean()
        add(
            name + '_ma7',
            trailing,
            trailing.index,
            'calendar_daily',
            cfg['gpr_release_lag_calendar_days'],
            20,
        )
    return panel, audit, pd.DataFrame(inventory)


def make_features(panel, cfg):
    """Create causal candidate features for the model notebooks.

    Only information available at the origin appears in a row: lagged and
    trailing gold statistics, carried external values, their ages/missingness,
    and deterministic calendar attributes. Gold-derived features are computed
    from observed quotes only (never labels after ``t``), and missing windows
    remain missing.
    """
    x = pd.DataFrame(index=panel.index)

    # Gold history: limited forward fill so short gaps become usable, while
    # long gaps remain missing rather than imputed.
    price = panel.y.ffill(limit=cfg['gold_feature_ffill_limit'])
    x['gold_latest'] = price
    x['gold_missing'] = panel.y.isna().astype(int)

    log_price = np.log(price.where(price > 0))
    for lag in [1, 2, 5, 10, 20, 60]:
        x[f'gold_lag_{lag}'] = price.shift(lag)
    for window in [5, 20, 60]:
        x[f'gold_ret_{window}'] = log_price.diff(window)
        x[f'gold_vol_{window}'] = log_price.diff().rolling(
            window, min_periods=window
        ).std()
        x[f'gold_mean_ratio_{window}'] = (
            price / price.rolling(window, min_periods=window).mean() - 1
        )

    # External levels, missingness, and staleness age.
    roles = [
        'brent',
        'brent_spot',
        'wti',
        'yield10',
        'dxy',
        'fed',
        'cpi_yoy',
        'cpi_mom',
        'gpr',
        'gpr_acts',
        'gpr_threats',
        'gpr_ma7',
        'gpr_acts_ma7',
        'gpr_threats_ma7',
    ]
    for name in roles:
        x[name] = panel[name]
        x[name + '_missing'] = panel[name].isna().astype(int)
        x[name + '_age_days'] = panel[name + '_age_days']

    # Trailing changes: log differences for positive price series, absolute
    # differences for rates/indices (and negative WTI prices).
    for name in ['brent', 'brent_spot', 'dxy']:
        for lag in [5, 20]:
            x[f'{name}_ret_{lag}'] = np.log(
                panel[name].where(panel[name] > 0)
            ).diff(lag)
    for name in ['wti', 'yield10', 'fed']:
        for lag in [5, 20]:
            x[f'{name}_chg_{lag}'] = panel[name].diff(lag)

    # Policy (yield minus Fed) and calendar features known at the origin.
    x['yield_spread'] = panel.yield10 - panel.fed
    x['weekday'] = x.index.dayofweek
    x['year_sin'] = np.sin(2 * np.pi * x.index.dayofyear / 365.25)
    x['year_cos'] = np.cos(2 * np.pi * x.index.dayofyear / 365.25)
    return x.replace([np.inf, -np.inf], np.nan)


def feature_columns(x, group, model):
    """Pick the candidate feature set for a model family and feature group.

    Group meanings
    --------------
    - ``gold_only``: a model must rely purely on gold history/calendar inputs.
    - ``external``: adds the preselected external variables.

    Models that require sequences or fixed column order trim to compact sets;
    Prophet never receives gold-derived regressors because future gold values
    are unknown at an origin.
    """
    if model == 'prophet':
        gold = []
        extra = [
            'dxy',
            'fed',
            'yield10',
            'cpi_yoy',
            'gpr_ma7',
            'brent_spot',
            'wti',
        ]
    elif model in ('gru', 'lstm'):
        gold = [
            'gold_latest',
            'gold_missing',
            'gold_ret_5',
            'gold_ret_20',
            'gold_vol_20',
            'year_sin',
            'year_cos',
            'weekday',
        ]
        extra = [
            'dxy',
            'fed',
            'yield10',
            'cpi_yoy',
            'cpi_mom',
            'gpr_ma7',
            'gpr_acts_ma7',
            'gpr_threats_ma7',
            'brent_spot',
            'wti',
            'yield_spread',
            'dxy_age_days',
            'gpr_ma7_age_days',
            'wti_missing',
        ]
    else:
        gold = [
            c
            for c in x
            if c.startswith('gold_') or c in ('weekday', 'year_sin', 'year_cos')
        ]
        extra = [
            'dxy',
            'fed',
            'yield10',
            'cpi_yoy',
            'cpi_mom',
            'gpr_ma7',
            'gpr_acts_ma7',
            'gpr_threats_ma7',
            'brent_spot',
            'wti',
            'yield_spread',
            'dxy_age_days',
            'gpr_ma7_age_days',
            'wti_missing',
        ]
    return gold if group == 'gold_only' else gold + extra


def make_folds(index, cfg):
    """Build chronological evaluation folds from one panel index.

    Development folds are drawn from the pre-holdout data; each fold trains on
    history through ``cutoff`` and is scored on the naturally available labels
    between ``cutoff`` and ``stop``. The untouched holdout is appended last.
    """
    development_end = len(index) - cfg['holdout_steps'] - 1
    folds = []
    for fold in range(cfg['n_folds']):
        stop = development_end - (cfg['n_folds'] - 1 - fold) * cfg['validation_steps']
        cutoff = stop - cfg['validation_steps']
        if cutoff + 1 < cfg['min_train_steps']:
            raise ValueError('Insufficient history for configured folds.')
        folds.append({'fold': f'cv_{fold+1}', 'cutoff': cutoff, 'stop': stop})
    folds.append({'fold': 'holdout', 'cutoff': development_end, 'stop': len(index) - 1})
    return pd.DataFrame(folds)


def forecast_origins(panel, cutoff, stop, cfg):
    """Choose fixed origins inside an evaluation block.

    Requirements
    ------------
    - Origins must be observed gold quotes (fixed-origin forecasts reuse the
      last price as an anchor).
    - Every configured horizon must fit entirely inside the block so labels
      can be scored without leaking into the next fold.
    - ``origin_stride`` selects every Nth origin to limit overlap.
    """
    last = stop - max(cfg['horizons'])
    origins = np.arange(cutoff + 1, last + 1, cfg['origin_stride'])
    return origins[panel.y.iloc[origins].notna().to_numpy()]


def metrics(predictions):
    """Score forecast rows per (model, fold, horizon) against persistence.

    Only rows whose labels are observed are scored. The persistence baseline
    repeats the origin price; skill is the fraction of baseline MAE removed.
    """
    records = []
    for (model, fold, horizon), group in predictions.groupby(
        ['model', 'fold', 'horizon']
    ):
        valid = group.dropna(subset=['actual', 'prediction', 'baseline'])
        if valid.empty:
            continue
        error = valid.prediction - valid.actual
        mae = error.abs().mean()
        baseline_mae = (valid.baseline - valid.actual).abs().mean()
        records.append(
            {
                'model': model,
                'fold': fold,
                'horizon': horizon,
                'n': len(valid),
                'MAE': mae,
                'RMSE': np.sqrt(np.mean(error**2)),
                'MAPE': (error.abs() / valid.actual.abs()).mean() * 100,
                'MASE': (error.abs() / valid.mase_scale).mean(),
                'baseline_MAE': baseline_mae,
                'skill_vs_persistence': (
                    1 - mae / baseline_mae if baseline_mae else np.nan
                ),
            }
        )
    return pd.DataFrame(records)


def predictions_frame(panel, origins, values, model, fold, cutoff, cfg):
    """Attach actuals, persistence, drift, and the MASE scale to one forecast.
    """
    # MASE scale: the mean absolute first difference before each cutoff, so
    # the model skill is always relative positive-change error, per fold.
    differences = panel.y.iloc[:cutoff+1].diff().abs()
    scale = differences.mean()
    rows = []
    for origin, row in zip(origins, values):
        history = panel.y.iloc[max(0, origin-252):origin+1].dropna()
        elapsed = (
            panel.index.get_loc(history.index[-1])
            - panel.index.get_loc(history.index[0])
        )
        slope = (
            (history.iloc[-1] - history.iloc[0]) / elapsed if elapsed else 0.0
        )
        for horizon, value in zip(cfg['horizons'], row):
            target = origin + horizon
            rows.append(
                {
                    'model': model,
                    'fold': fold,
                    'origin': panel.index[origin],
                    'target_date': panel.index[target],
                    'horizon': horizon,
                    'actual': panel.y.iloc[target],
                    'prediction': float(value),
                    'baseline': panel.y.iloc[origin],
                    'drift': panel.y.iloc[origin] + horizon * slope,
                    'mase_scale': scale,
                }
            )
    return pd.DataFrame(rows)


def seed_everything(seed):
    """Reset random seeds for reproducible model-fitting runs."""
    random.seed(seed)
    np.random.seed(seed)


def validate_panel(panel, x):
    """Assert the panel/candidate contract before any model sees the data."""
    assert panel.index.is_unique and panel.index.is_monotonic_increasing
    assert (panel.index.dayofweek < 5).all()
    assert x.index.equals(panel.index)
    assert not np.isinf(x.to_numpy()).any()
    assert panel.loc[panel.target_observed.eq(1), 'y'].notna().all()


def preprocess(cfg=None):
    """Run the full shared preprocessing and cache everything to disk."""
    cfg = cfg or load_config()
    output = ROOT / 'data/processed'
    output.mkdir(parents=True, exist_ok=True)
    panel, audit, inventory = build_panel(ROOT / 'data/raw', cfg)
    x = make_features(panel, cfg)
    validate_panel(panel, x)

    panel.to_csv(output / 'panel.csv', index_label='ds')
    x.to_csv(output / 'features.csv', index_label='ds')
    audit.to_csv(output / 'gold_cleaning_audit.csv', index=False)
    inventory.to_csv(output / 'availability_assumptions.csv', index=False)
    make_folds(panel.index, cfg).to_csv(output / 'folds.csv', index=False)
    pd.DataFrame(
        {
            'feature': x.columns,
            'meaning': [feature_description(c) for c in x],
        }
    ).to_csv(output / 'feature_dictionary.csv', index=False)
    (output / 'config_snapshot.json').write_text(json.dumps(cfg, indent=2))

    manifest = [
        {
            'file': path.name,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted((ROOT / 'data/raw').iterdir())
        if path.is_file()
    ]
    pd.DataFrame(manifest).to_csv(output / 'source_manifest.csv', index=False)
    return panel, x, audit, inventory


def feature_description(name):
    """Return a short human-readable description for one candidate feature."""
    if name.endswith('_age_days'):
        return (
            'Calendar days since the latest source observation; '
            'unavailable until assumed release.'
        )
    if 'missing' in name:
        return 'Indicator: source value unavailable at this origin.'
    if '_ret_' in name:
        return 'Trailing log difference over the indicated weekday-grid steps.'
    if '_chg_' in name:
        return 'Trailing absolute difference over the indicated weekday-grid steps.'
    if name.startswith('gold_'):
        return (
            'Gold history through origin t; filled inputs limited by '
            'configuration, labels never filled.'
        )
    return 'Origin-time value or deterministic calendar feature; see README and availability table.'


def load_prepared():
    """Load the cached processed data from notebook 01 and verify freshness.

    The local config must match the config.json snapshot used during
    preprocessing. Raw-file hashes must still match the manifest, otherwise
    the pipeline must be rerun before any model training.
    """
    out = ROOT / 'data/processed'
    panel = pd.read_csv(out / 'panel.csv', index_col='ds', parse_dates=True)
    x = pd.read_csv(out / 'features.csv', index_col='ds', parse_dates=True)
    snapshot = json.loads((out / 'config_snapshot.json').read_text())
    manifest = pd.read_csv(out / 'source_manifest.csv')

    # Every raw file must be the same one used when notebook 01 was run.
    for row in manifest.itertuples(index=False):
        source = ROOT / 'data/raw' / row.file
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if not source.exists() or digest != row.sha256:
            raise ValueError('Raw data changed: rerun notebook 01 and rebuild model artifacts.')

    cfg = load_config()
    ignored = {'final_holdout_enabled', 'max_epochs', 'patience'}
    differing = [key for key in cfg if key not in ignored and snapshot[key] != cfg[key]]
    if differing:
        raise ValueError('Configuration changed: rerun notebook 01 before training.')
    if not panel.index.equals(x.index):
        raise ValueError('Processed panel and feature dates differ: rerun notebook 01.')
    validate_panel(panel, x)
    return panel, x, pd.read_csv(out / 'folds.csv')
