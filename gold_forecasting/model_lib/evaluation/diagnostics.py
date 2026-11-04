"""Shared diagnostic metrics for experimental notebook recipes, not serving."""

from typing import TypedDict

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


class PointErrors(TypedDict):
    """Point error summaries in price units (MAPE is percentage points)."""

    MAPE: float
    RMSE: float
    MAE: float


def error_metrics(actual: ArrayLike, predicted: ArrayLike) -> PointErrors:
    """Compute legacy point diagnostics; caller owns split/availability correctness."""
    actual_array = np.asarray(actual, dtype=float)
    prediction_array = np.asarray(predicted, dtype=float)
    error = prediction_array - actual_array
    return {
        "MAPE": float(np.mean(np.abs(error / actual_array)) * 100),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(np.abs(error))),
    }


def interval_diagnostics(
    actual: pd.Series, lower: pd.Series, upper: pd.Series
) -> dict[str, float | int]:
    """Return empirical coverage/width on observed complete rows, without calibration claims."""
    frame = pd.concat(
        [actual.rename("actual"), lower.rename("lower"), upper.rename("upper")], axis=1
    ).dropna()
    if frame.empty or (frame.lower > frame.upper).any():
        raise ValueError("Intervals need observed targets and ordered finite bounds.")
    return {
        "n": len(frame),
        "coverage": float(
            ((frame.actual >= frame.lower) & (frame.actual <= frame.upper)).mean()
        ),
        "mean_width": float((frame.upper - frame.lower).mean()),
    }
