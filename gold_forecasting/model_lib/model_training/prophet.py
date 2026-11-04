"""Gold model training / prophet. Extracted default daily algorithms; see function contracts.

Model-fitting workflows are invoked only by notebooks, never by serving.
"""

from __future__ import annotations


import numpy as np
import pandas as pd
from model_lib.contracts import Config, Bundle


def fit_prophet(
    panel: pd.DataFrame, x: pd.DataFrame, cutoff: int, params: Config, cfg: Config
) -> Bundle:
    """Fit Prophet on log gold prices with optional external regressors."""
    from model_lib.preprocessing.features import feature_columns
    from prophet import Prophet

    columns = feature_columns(x, params["group"], "prophet")
    train = pd.DataFrame(
        {
            "ds": panel.index[: cutoff + 1],
            "y": panel.y.iloc[: cutoff + 1].to_numpy(),
        }
    )
    # Log scale preserves positivity and lets trend changes act proportionally.
    train["y"] = np.log(train.y)
    medians = x.iloc[: cutoff + 1][columns].median().fillna(0)
    for column in columns:
        train[column] = x.iloc[: cutoff + 1][column].fillna(medians[column]).to_numpy()
    model = Prophet(
        changepoint_prior_scale=params["changepoint_prior_scale"],
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
    model.fit(train[["ds", "y"] + kept_columns].dropna(subset=["y"]))
    return {
        "kind": "prophet",
        "model": model,
        "columns": kept_columns,
        "medians": medians,
        "params": params,
    }
