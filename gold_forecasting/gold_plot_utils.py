"""Readable plotting helpers for the gold forecasting notebooks.

One module keeps every forecast plot consistent across the model notebooks
and the model-comparison notebook. Functions here do not change prediction
logic; they only inspect the already-generated prediction tables.
"""
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# One-row lookahead: each development fold has its training data immediately
# before its test block. A 252-weekday context window is a full trading year.
TRAIN_CONTEXT_STEPS = 252

# Consistent colors for the small number of series used by every plot.
MODEL_KINDS = ('prophet', 'xgboost', 'gru', 'lstm')
BASELINE_KINDS = ('persistence', 'trailing_drift')
MODEL_COLORS = {
    'prophet': '#1f77b4',
    'xgboost': '#ff7f0e',
    'gru': '#2ca02c',
    'lstm': '#9467bd',
    'persistence': '#7f7f7f',
    'trailing_drift': '#a0522d',
}
HORIZON_COLORS = {
    5: '#1f77b4',
    21: '#ff7f0e',
    63: '#2ca02c',
    126: '#d62728',
    252: '#9467bd',
}
TEST_ALPHA = 0.10


def _ensure_plots_dir(out_dir):
    """Create and return the per-model or comparison `plots/` directory."""
    directory = Path(out_dir) / 'plots'
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _show(fig, path):
    """Draw and save a matplotlib figure without leaking figure objects.

    A non-interactive backend (e.g. Agg) skips display, because no window
    manager is available in that context.
    """
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def _as_index(panel, offset):
    """Return the panel timestamp at a zero-based integer row offset."""
    return panel.index[int(offset)]


def _dev_folds(folds):
    """Return only the chronological development folds.

   词典 folds has `cutoff` and `stop` as integer panel-row offsets. The
    development folds are ordered by their starting index.
    """
    rows = folds.loc[folds.fold.ne('holdout')].copy()
    if not rows.empty:
        rows = rows.sort_values('cutoff')
    return rows


def _choose_folds(folds, stage='cv'):
    """Select the folds relevant to `stage`.

    `cv` returns the development folds.
    `holdout` returns only the untouched holdout row, if any.
    """
    if stage == 'cv':
        return _dev_folds(folds)
    if stage == 'holdout':
        return folds.loc[folds.fold.eq('holdout')]
    raise ValueError(f'Unknown stage: {stage}')


def _panel_range_for_folds(panel, folds, stage='cv'):
    """Return the panel index range covered by a set of evaluation folds."""
    if len(folds) == 0:
        return None
    start = min(int(fold.cutoff) for fold in folds.itertuples())
    stop = max(int(fold.stop) for fold in folds.itertuples())
    # `stop` is inclusive in the original data, so we need +1 to slice correctly.
    return panel.index[start:stop + 1]


def _mark_cutoff(ax, panel, fold):
    """Draw a vertical line at the training/test boundary of one fold."""
    cutoff_date = _as_index(panel, fold.cutoff)
    ax.axvline(
        cutoff_date,
        color='black',
        linestyle='--',
        linewidth=1,
        label='Train/test cutoff',
    )


def _shade_test_block(ax, panel, fold, alpha=TEST_ALPHA, label=None):
    """Shade only the outer-test window of one fold."""
    test_start = _as_index(panel, fold.cutoff + 1)
    test_end = _as_index(panel, fold.stop)
    ax.axvspan(test_start, test_end, color='tab:blue', alpha=alpha)


def _plot_actual_line(ax, panel, label='Actual gold price'):
    """Draw the observed gold series on one axis."""
    ax.plot(
        panel.index,
        panel.y,
        color='#151515',
        linewidth=0.9,
        label=label,
    )


def _forecast_color(model):
    """Return a stable color for a model kind."""
    return MODEL_COLORS.get(model, '#8c564b')


def _forecast_color_by_horizon(horizon):
    """Return a stable color for one configured horizon."""
    return HORIZON_COLORS.get(horizon, '#17becf')


def plot_model_cv(panel, predictions, folds, kind, cfg, out_dir):
    """Save and display four development-stage plots for one model.

    Predictions are plotted against the observed gold price. Panels are:
    1. full observed history plus every outer prediction;
    2. one panel per horizon with outer predictions;
    3. one panel per fold with training context plus test predictions;
    4. one panel per horizon of actual vs predicted prices.
    """
    plot_dir = _ensure_plots_dir(out_dir)
    scores = predictions.dropna(subset=['actual', 'prediction']).copy()
    if scores.empty:
        print('No observed development predictions to plot yet.')
        return []
    dev = _dev_folds(folds)
    horizons = sorted(scores.horizon.unique())
    paths = []

    # 1. Full history with every development prediction.
    path = plot_dir / 'actual_and_forecasts_timeline.png'
    fig, ax = plt.subplots(figsize=(14, 6))
    _plot_actual_line(ax, panel)
    for fold in dev.itertuples(index=False):
        _shade_test_block(ax, panel, fold)
    for horizon in horizons:
        block = scores.loc[scores.horizon.eq(horizon)]
        ax.scatter(
            block.target_date,
            block.prediction,
            s=12,
            alpha=0.75,
            color=_forecast_color_by_horizon(horizon),
            label=f'Forecast h={horizon}',
        )
    ax.set_title(f'{kind}: observed prices and development forecasts')
    ax.set_ylabel('USD/oz')
    ax.legend(frameon=False)
    _show(fig, path)
    paths.append(path)

    # 2. Actual vs forecast for each configured horizon.
    path = plot_dir / 'actual_and_forecasts_by_horizon.png'
    fig, axes = plt.subplots(
        len(horizons),
        1,
        figsize=(14, 3.4 * len(horizons)),
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    for ax, horizon in zip(axes, horizons):
        _plot_actual_line(ax, panel)
        for fold in dev.itertuples(index=False):
            _shade_test_block(ax, panel, fold)
        block = scores.loc[scores.horizon.eq(horizon)]
        ax.scatter(
            block.target_date,
            block.prediction,
            s=14,
            alpha=0.85,
            color=_forecast_color_by_horizon(horizon),
            label=f'{kind} forecast',
        )
        ax.set_ylabel('USD/oz')
        ax.set_title(f'Horizon {horizon} development forecasts')
        ax.legend(frameon=False)
    paths.append(path)
    _show(fig, path)

    # 3. One panel per fold: training context, test actual, and forecasts.
    path = plot_dir / 'actual_and_forecasts_by_fold.png'
    fig, axes = plt.subplots(
        len(dev),
        1,
        figsize=(14, 3.0 * len(dev)),
        sharex=False,
    )
    axes = np.atleast_1d(axes)
    for ax, fold in zip(axes, dev.itertuples(index=False)):
        cutoff = int(fold.cutoff)
        stop = int(fold.stop)
        start = max(0, cutoff - TRAIN_CONTEXT_STEPS)
        end = stop + 1
        ax.plot(
            panel.index[start:end],
            panel.y.iloc[start:end],
            color='#151515',
            linewidth=0.9,
            label='Actual gold price',
        )
        _shade_test_block(ax, panel, fold)
        _mark_cutoff(ax, panel, fold)
        block = scores.loc[scores.fold.eq(fold.fold)]
        for horizon in horizons:
            horizon_data = block.loc[block.horizon.eq(horizon)]
            if horizon_data.empty:
                continue
            ax.scatter(
                horizon_data.target_date,
                horizon_data.prediction,
                s=14,
                alpha=0.85,
                color=_forecast_color_by_horizon(horizon),
                label=f'Forecast h={horizon}',
            )
        ax.set_title(f'{kind}: {fold.fold}')
        ax.set_ylabel('USD/oz')
        ax.legend(frameon=False, fontsize=8)
    paths.append(path)
    _show(fig, path)

    # 4. Actual vs predicted price, one panel per horizon.
    path = plot_dir / 'actual_vs_predicted_scatter.png'
    fig, axes = plt.subplots(
        len(horizons),
        1,
        figsize=(5.0 * len(horizons), 4.0 * len(horizons)),
        sharex=False,
    )
    axes = np.atleast_1d(axes)
    for ax, horizon in zip(axes, horizons):
        block = scores.loc[scores.horizon.eq(horizon)]
        low = min(block.actual.min(), block.prediction.min())
        high = max(block.actual.max(), block.prediction.max())
        pad = 0.05 * (high - low or 1.0)
        ax.plot(
            [low - pad, high + pad],
            [low - pad, high + pad],
            color='black',
            linestyle='--',
            linewidth=1,
            label='Perfect forecast',
        )
        ax.scatter(
            block.actual,
            block.prediction,
            s=14,
            alpha=0.70,
            color=_forecast_color_by_horizon(horizon),
            label=f'{kind} forecasts',
        )
        ax.set_xlabel('Actual price (USD/oz)')
        ax.set_ylabel('Predicted price (USD/oz)')
        ax.set_title(f'Actual vs predicted: horizon {horizon}')
        ax.legend(frameon=False)
    paths.append(path)
    _show(fig, path)
    return paths


def plot_model_holdout(panel, predictions, folds, kind, cfg, out_dir):
    """Save and display the holdout-stage plots for one model."""
    plot_dir = _ensure_plots_dir(out_dir)
    scores = predictions.dropna(subset=['actual', 'prediction']).copy()
    if scores.empty:
        print('No observed holdout predictions to plot yet.')
        return []
    holdout_rows = _choose_folds(folds, stage='holdout')
    if len(holdout_rows) == 0:
        print('No holdout fold configured in folds.csv.')
        return []
    fold = holdout_rows.iloc[0]
    cutoff = int(fold.cutoff)
    stop = int(fold.stop)
    start = max(0, cutoff - TRAIN_CONTEXT_STEPS)
    end = stop + 1
    paths = []

    path = plot_dir / 'holdout_actual_and_forecasts.png'
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(
        panel.index[start:end],
        panel.y.iloc[start:end],
        color='#151515',
        linewidth=0.9,
        label='Actual gold price',
    )
    _shade_test_block(ax, panel, fold)
    _mark_cutoff(ax, panel, fold)
    for horizon in sorted(scores.horizon.unique()):
        block = scores.loc[scores.horizon.eq(horizon)]
        ax.scatter(
            block.target_date,
            block.prediction,
            s=16,
            alpha=0.85,
            color=_forecast_color_by_horizon(horizon),
            label=f'Forecast h={horizon}',
        )
    ax.set_title(f'{kind}: holdout forecast vs observed prices')
    ax.set_ylabel('USD/oz')
    ax.legend(frameon=False)
    paths.append(path)
    _show(fig, path)

    horizons = sorted(scores.horizon.unique())
    path = plot_dir / 'holdout_actual_vs_predicted_scatter.png'
    fig, axes = plt.subplots(
        len(horizons),
        1,
        figsize=(5.0 * len(horizons), 4.0 * len(horizons)),
        sharex=False,
    )
    axes = np.atleast_1d(axes)
    for ax, horizon in zip(axes, horizons):
        block = scores.loc[scores.horizon.eq(horizon)]
        ax.scatter(
            block.actual,
            block.prediction,
            s=16,
            alpha=0.70,
            color=_forecast_color_by_horizon(horizon),
            label=f'{kind} forecasts',
        )
        ax.set_xlabel('Actual price (USD/oz)')
        ax.set_ylabel('Predicted price (USD/oz)')
        ax.set_title(f'Holdout actual vs predicted: horizon {horizon}')
        ax.legend(frameon=False)
    paths.append(path)
    _show(fig, path)
    return paths


def plot_future_forecast(panel, forecast, kind, cfg, out_dir):
    """Plot the final forecast path against the observed history.

    The final forecast has no future actual prices; this plot is a path
    visual, not a validated accuracy result.
    """
    plot_dir = _ensure_plots_dir(out_dir)
    if 'forecast' not in forecast.columns:
        print('No future forecast column available yet.')
        return []
    observed = forecast.dropna(subset=['forecast']).copy()
    if observed.empty:
        print('No future forecast lines to plot yet.')
        return []
    origin_date = pd.to_datetime(observed.origin.max())
    start = max(0, len(panel) - TRAIN_CONTEXT_STEPS)
    path = plot_dir / 'future_forecast.png'
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(
        panel.index[start:],
        panel.y.iloc[start:],
        color='#151515',
        linewidth=0.9,
        label='Observed gold price',
    )
    ax.axvline(origin_date, color='black', linestyle='--', linewidth=1, label='Forecast origin')
    for horizon in sorted(observed.horizon.unique()):
        block = observed.loc[observed.horizon.eq(horizon)]
        ax.scatter(
            block.target_date,
            block.forecast,
            s=22,
            color=_forecast_color_by_horizon(horizon),
            label=f'Forecast h={horizon}',
        )
    ax.set_title(f'{kind}: locked forecast into the future')
    ax.set_ylabel('USD/oz')
    ax.legend(frameon=False)
    paths = [path]
    _show(fig, path)
    return paths


def _mean_by_target_date(df, horizon=None):
    """Aggregate overlapping forecasts to one mean value per target date."""
    frame = df.dropna(subset=['actual', 'prediction']).copy()
    if horizon is not None:
        frame = frame.loc[frame.horizon.eq(horizon)]
    if frame.empty:
        return pd.DataFrame(columns=['model', 'target_date', 'prediction'])
    grouped = (
        frame.groupby(['model', 'target_date'], as_index=False)
        .prediction.mean()
    )
    return grouped


def plot_comparison_actual_and_forecasts(
    predictions, panel, folds, cfg, out_dir, stage='cv'
):
    """Plot mean forecast paths for every model on common observable targets."""
    plot_dir = _ensure_plots_dir(out_dir)
    if predictions.empty:
        print('No comparison predictions to plot yet.')
        return []
    fold_rows = _choose_folds(folds, stage=stage)
    if len(fold_rows) == 0:
        print('No fold rows available for the requested stage.')
        return []
    horizons = sorted(predictions.dropna(subset=['horizon']).horizon.unique())
    paths = []
    stage_word = 'development' if stage == 'cv' else 'holdout'

    path = plot_dir / f'comparison_actual_and_forecasts.png'
    fig, axes = plt.subplots(
        len(horizons),
        1,
        figsize=(14, 3.4 * len(horizons)),
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    for ax, horizon in zip(axes, horizons):
        _plot_actual_line(ax, panel)
        for fold in fold_rows.itertuples(index=False):
            _shade_test_block(ax, panel, fold)
        models = [m for m in predictions.model.unique() if m in MODEL_KINDS]
        baselines = [b for b in predictions.model.unique() if b in BASELINE_KINDS]
        for model in models:
            grouped = _mean_by_target_date(predictions, horizon=horizon)
            model_frame = grouped.loc[grouped.model.eq(model)]
            if model_frame.empty:
                continue
            ax.plot(
                model_frame.target_date,
                model_frame.prediction,
                color=_forecast_color(model),
                linewidth=1.4,
                marker='o',
                markersize=3,
                label=f'{model} (mean)',
            )
        for model in baselines:
            grouped = _mean_by_target_date(predictions, horizon=horizon)
            model_frame = grouped.loc[grouped.model.eq(model)]
            if model_frame.empty:
                continue
            ax.plot(
                model_frame.target_date,
                model_frame.prediction,
                color=_forecast_color(model),
                linewidth=1.0,
                linestyle='--',
                label=f'{model} baseline',
            )
        ax.set_ylabel('USD/oz')
        ax.set_title(f'{stage_word.title()} comparison: horizon {horizon}')
        ax.legend(frameon=False, fontsize=8)
    paths.append(path)
    _show(fig, path)
    return paths


def plot_comparison_by_fold(
    predictions, panel, folds, cfg, out_dir, stage='cv'
):
    """Plot the longest horizon per fold with one mean line per model."""
    plot_dir = _ensure_plots_dir(out_dir)
    if predictions.empty:
        print('No comparison predictions to plot yet.')
        return []
    fold_rows = _choose_folds(folds, stage=stage)
    if len(fold_rows) == 0:
        print('No fold rows available for the requested stage.')
        return []
    horizon = max(predictions.dropna(subset=['horizon']).horizon.unique())
    paths = []
    stage_word = 'development' if stage == 'cv' else 'holdout'
    path = plot_dir / f'comparison_by_fold.png'
    fig, axes = plt.subplots(
        len(fold_rows),
        1,
        figsize=(14, 3.0 * len(fold_rows)),
        sharex=False,
    )
    axes = np.atleast_1d(axes)
    for ax, fold in zip(axes, fold_rows.itertuples(index=False)):
        cutoff = int(fold.cutoff)
        stop = int(fold.stop)
        start = max(0, cutoff - TRAIN_CONTEXT_STEPS)
        end = stop + 1
        ax.plot(
            panel.index[start:end],
            panel.y.iloc[start:end],
            color='#151515',
            linewidth=0.9,
            label='Actual gold price',
        )
        _shade_test_block(ax, panel, fold)
        _mark_cutoff(ax, panel, fold)
        models = [m for m in predictions.model.unique() if m in MODEL_KINDS]
        baselines = [b for b in predictions.model.unique() if b in BASELINE_KINDS]
        grouped = _mean_by_target_date(predictions, horizon=horizon)
        for model in models:
            model_frame = grouped.loc[
                grouped.model.eq(model)
                & grouped.target_date.between(
                    _as_index(panel, cutoff + 1),
                    _as_index(panel, stop),
                )
            ]
            if model_frame.empty:
                continue
            ax.plot(
                model_frame.target_date,
                model_frame.prediction,
                color=_forecast_color(model),
                linewidth=1.4,
                marker='o',
                markersize=3,
                label=f'{model} (mean)',
            )
        for model in baselines:
            model_frame = grouped.loc[
                grouped.model.eq(model)
                & grouped.target_date.between(
                    _as_index(panel, cutoff + 1),
                    _as_index(panel, stop),
                )
            ]
            if model_frame.empty:
                continue
            ax.plot(
                model_frame.target_date,
                model_frame.prediction,
                color=_forecast_color(model),
                linewidth=1.0,
                linestyle='--',
                label=f'{model} baseline',
            )
        ax.set_title(f'{stage_word.title()} comparison: {fold.fold}, horizon {horizon}')
        ax.set_ylabel('USD/oz')
        ax.legend(frameon=False, fontsize=8)
    paths.append(path)
    _show(fig, path)
    return paths


def plot_comparison_scatter(
    predictions, panel, folds, cfg, out_dir, stage='cv'
):
    """Plot actual vs predicted price across all models, one panel per horizon."""
    plot_dir = _ensure_plots_dir(out_dir)
    if predictions.empty:
        print('No comparison predictions to plot yet.')
        return []
    horizons = sorted(predictions.dropna(subset=['horizon']).horizon.unique())
    paths = []
    stage_word = 'development' if stage == 'cv' else 'holdout'
    path = plot_dir / f'comparison_actual_vs_predicted_scatter.png'
    fig, axes = plt.subplots(
        len(horizons),
        1,
        figsize=(5.0 * len(horizons), 4.0 * len(horizons)),
        sharex=False,
    )
    axes = np.atleast_1d(axes)
    for ax, horizon in zip(axes, horizons):
        frame = predictions.loc[
            predictions.horizon.eq(horizon)
        ].dropna(subset=['actual', 'prediction'])
        low = min(frame.actual.min(), frame.prediction.min())
        high = max(frame.actual.max(), frame.prediction.max())
        pad = 0.05 * (high - low or 1.0)
        ax.plot(
            [low - pad, high + pad],
            [low - pad, high + pad],
            color='black',
            linestyle='--',
            linewidth=1,
            label='Perfect forecast',
        )
        models = [m for m in frame.model.unique() if m in MODEL_KINDS]
        baselines = [b for b in frame.model.unique() if b in BASELINE_KINDS]
        for model in models:
            block = frame.loc[frame.model.eq(model)]
            ax.scatter(
                block.actual,
                block.prediction,
                s=14,
                alpha=0.65,
                color=_forecast_color(model),
                label=f'{model}',
            )
        for model in baselines:
            block = frame.loc[frame.model.eq(model)]
            ax.scatter(
                block.actual,
                block.prediction,
                s=14,
                alpha=0.65,
                color=_forecast_color(model),
                marker='x',
                label=f'{model} baseline',
            )
        ax.set_xlabel('Actual price (USD/oz)')
        ax.set_ylabel('Predicted price (USD/oz)')
        ax.set_title(f'{stage_word.title()} comparison: horizon {horizon}')
        ax.legend(frameon=False, fontsize=8)
    paths.append(path)
    _show(fig, path)
    return paths


def plot_comparison_error_ranking(
    predictions, cfg, out_dir, stage='cv'
):
    """Plot overall and per-horizon MAE ranking across models."""
    plot_dir = _ensure_plots_dir(out_dir)
    scores = predictions.dropna(subset=['actual', 'prediction']).copy()
    if scores.empty:
        print('No comparison predictions to rank yet.')
        return []
    stage_word = 'development' if stage == 'cv' else 'holdout'
    paths = []
    path = plot_dir / f'comparison_error_ranking.png'
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    scores['absolute_error'] = (scores.prediction - scores.actual).abs()
    overall = scores.groupby('model').absolute_error.mean().sort_values()
    by_horizon = (
        scores.groupby(['model', 'horizon']).absolute_error.mean().unstack()
    )
    overall.plot.bar(ax=axes[0], color='#1f77b4')
    axes[0].set_title(f'{stage_word.title()} overall MAE')
    axes[0].set_ylabel('MAE (USD/oz)')
    axes[0].tick_params(axis='x', rotation=45)
    by_horizon.plot.bar(ax=axes[1])
    axes[1].set_title(f'{stage_word.title()} MAE by horizon')
    axes[1].set_ylabel('MAE (USD/oz)')
    axes[1].legend(title='Horizon', frameon=False)
    paths.append(path)
    _show(fig, path)
    return paths
