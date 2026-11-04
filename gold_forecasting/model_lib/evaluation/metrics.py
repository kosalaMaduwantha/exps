"""Observed-target point metrics and baseline attachment for notebook evaluation."""

import numpy as np
import pandas as pd

from model_lib.contracts import Config


def metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Score rows per model/fold/horizon, ignoring unobserved labels.

    Input columns: model, fold, horizon, actual, prediction, baseline, mase_scale.
    Returns n, MAE/RMSE/MAPE/MASE, baseline_MAE and skill_vs_persistence.
    """
    records = []
    for (model, fold, horizon), group in predictions.groupby(
        ["model", "fold", "horizon"]
    ):
        valid = group.dropna(subset=["actual", "prediction", "baseline"])
        if valid.empty:
            continue
        error = valid.prediction - valid.actual
        mae = error.abs().mean()
        baseline_mae = (valid.baseline - valid.actual).abs().mean()
        records.append(
            {
                "model": model,
                "fold": fold,
                "horizon": horizon,
                "n": len(valid),
                "MAE": mae,
                "RMSE": np.sqrt(np.mean(error**2)),
                "MAPE": (error.abs() / valid.actual.abs()).mean() * 100,
                "MASE": (error.abs() / valid.mase_scale).mean(),
                "baseline_MAE": baseline_mae,
                "skill_vs_persistence": 1 - mae / baseline_mae
                if baseline_mae
                else np.nan,
            }
        )
    return pd.DataFrame(records)


def predictions_frame(
    panel: pd.DataFrame,
    origins: np.ndarray,
    values: np.ndarray,
    model: str,
    fold: str,
    cutoff: int,
    cfg: Config,
) -> pd.DataFrame:
    """Attach observed actuals and origin-time persistence/drift to a price array.

    values is shaped (origin_count, horizon_count). MASE scale uses differences
    before the training cutoff. Targets must lie inside panel; missing labels
    remain missing and are excluded only when metrics are computed.
    """
    scale = panel.y.iloc[: cutoff + 1].diff().abs().mean()
    rows = []
    for origin, row in zip(origins, values):
        history = panel.y.iloc[max(0, origin - 252) : origin + 1].dropna()
        elapsed = panel.index.get_loc(history.index[-1]) - panel.index.get_loc(
            history.index[0]
        )
        slope = (history.iloc[-1] - history.iloc[0]) / elapsed if elapsed else 0.0
        for horizon, value in zip(cfg["horizons"], row):
            rows.append(
                {
                    "model": model,
                    "fold": fold,
                    "origin": panel.index[origin],
                    "target_date": panel.index[origin + horizon],
                    "horizon": horizon,
                    "actual": panel.y.iloc[origin + horizon],
                    "prediction": float(value),
                    "baseline": panel.y.iloc[origin],
                    "drift": panel.y.iloc[origin] + horizon * slope,
                    "mase_scale": scale,
                }
            )
    return pd.DataFrame(rows)
