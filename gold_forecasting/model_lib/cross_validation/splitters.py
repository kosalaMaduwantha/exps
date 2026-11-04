"""Chronological folds and fixed-origin evaluation eligibility; no model fitting."""

import numpy as np
import pandas as pd

from model_lib.contracts import Config


def make_folds(index: pd.DatetimeIndex, cfg: Config) -> pd.DataFrame:
    """Return fold/cutoff/stop rows with integer positions and an untouched holdout."""
    development_end = len(index) - cfg["holdout_steps"] - 1
    folds = []
    for fold in range(cfg["n_folds"]):
        stop = development_end - (cfg["n_folds"] - 1 - fold) * cfg["validation_steps"]
        cutoff = stop - cfg["validation_steps"]
        if cutoff + 1 < cfg["min_train_steps"]:
            raise ValueError("Insufficient history for configured folds.")
        folds.append({"fold": f"cv_{fold + 1}", "cutoff": cutoff, "stop": stop})
    folds.append({"fold": "holdout", "cutoff": development_end, "stop": len(index) - 1})
    return pd.DataFrame(folds)


def forecast_origins(
    panel: pd.DataFrame, cutoff: int, stop: int, cfg: Config
) -> np.ndarray:
    """Return observed positions after cutoff whose largest horizon fits before stop."""
    last = stop - max(cfg["horizons"])
    origins = np.arange(cutoff + 1, last + 1, cfg["origin_stride"])
    return origins[panel.y.iloc[origins].notna().to_numpy()]
