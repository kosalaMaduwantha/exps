"""Gold forecasting / regressors. Extracted default daily algorithms; see function contracts.

Model-fitting workflows are invoked only by notebooks, never by serving.
"""

from __future__ import annotations


import numpy as np
import pandas as pd
from model_lib.contracts import Config


def forecast_external_paths(
    panel: pd.DataFrame, x: pd.DataFrame, origin: int, columns: list[str], cfg: Config
) -> pd.DataFrame:
    """Forecast anchored future external regressor values with Prophet.

    Each regressor uses its own daily as-of history through ``origin``. The
    fitted coefficients of the main gold model stay frozen while the future
    macro inputs follow their own forecast paths. Paths are anchored so the
    first forecast step continues from the last observed regressor value.
    """
    from prophet import Prophet

    future_dates = pd.bdate_range(
        panel.index[origin] + pd.offsets.BDay(1),
        periods=max(cfg["horizons"]),
    )
    paths = pd.DataFrame({"ds": future_dates})
    for column in columns:
        history = x[column].iloc[: origin + 1].dropna()
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
        model.fit(pd.DataFrame({"ds": history.index, "y": history.to_numpy()}))
        predict_dates = pd.DatetimeIndex([history.index[-1]]).append(future_dates)
        forecast = model.predict(pd.DataFrame({"ds": predict_dates}))
        adjustment = history.iloc[-1] - forecast["yhat"].iloc[0]
        paths[column] = forecast["yhat"].iloc[1:].to_numpy() + adjustment
    return paths
