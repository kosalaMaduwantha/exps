"""Gold model training / xgboost. Extracted default daily algorithms; see function contracts.

Model-fitting workflows are invoked only by notebooks, never by serving.
"""

from __future__ import annotations


import numpy as np
import pandas as pd
from model_lib.contracts import Config, Bundle


def supervised_rows(
    panel: pd.DataFrame, cutoff: int, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return eligible training origins and their log-change labels.

    A training row is eligible only when both the origin and its target are
    observed, and the target date does not exceed the training cutoff. Labels
    are ``log(price_after_h / price_now)`` shifts, matching how the model
    predicts relative moves rather than raw levels.
    """
    origins = np.arange(60, cutoff - horizon + 1)
    valid = (
        panel.y.iloc[origins].notna().to_numpy()
        & panel.y.iloc[origins + horizon].notna().to_numpy()
    )
    origins = origins[valid]
    target = np.log(
        panel.y.iloc[origins + horizon].to_numpy() / panel.y.iloc[origins].to_numpy()
    )
    return origins, target


def fit_xgboost(
    panel: pd.DataFrame, x: pd.DataFrame, cutoff: int, params: Config, cfg: Config
) -> Bundle:
    """Train one tree model per horizon on rows inside the training cutoff."""
    from model_lib.preprocessing.features import feature_columns
    from xgboost import XGBRegressor

    columns = feature_columns(x, params["group"], "xgboost")
    models = []
    for horizon in cfg["horizons"]:
        rows, target = supervised_rows(panel, cutoff, horizon)
        if len(rows) < 100:
            raise ValueError("Fewer than 100 eligible training examples.")
        model = XGBRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,
            reg_lambda=5,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=2,
            random_state=cfg["seed"],
        )
        # Native missing-value handling: no fitted scaler is needed. Boosting
        # rounds are fixed by the inner validation search, not outer stopping.
        model.fit(x.iloc[rows][columns], target)
        models.append(model)
    return {
        "kind": "xgboost",
        "models": models,
        "columns": columns,
        "params": params,
    }
