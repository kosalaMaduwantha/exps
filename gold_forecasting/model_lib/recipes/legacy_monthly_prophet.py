"""Named diagnostic recipe extracted from prophet.ipynb.

Original algorithms are preserved with explicit context rather than notebook globals.
These experiments may use realized future regressors, full-series screening, or
holdout-driven tuning. They must not publish deployable accuracy/selection claims.
Any annotations mark heterogeneous research/backend results or raw source values.
Use validated_daily for API deployments. No work runs at import time.
"""

from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd


def month_start(s: Any, dayfirst: bool = False) -> pd.Series:
    """Normalize any date-like input to the first day of its calendar month.

    Notebook context is explicit keyword-only input: none.
    Returns: pd.Series.
    This recipe is diagnostic-only; oracle regressors/holdout tuning are not deployable selection."""
    d = pd.to_datetime(pd.Series(s), dayfirst=dayfirst)
    return d.dt.to_period("M").dt.to_timestamp()


def monthly_mean(
    df: pd.DataFrame, date_col: str, value_col: str, **mw: Any
) -> pd.Series:
    """Aggregate a (possibly daily) series to calendar-month means.

    Notebook context is explicit keyword-only input: none.
    Returns: pd.Series.
    This recipe is diagnostic-only; oracle regressors/holdout tuning are not deployable selection."""
    d = df.dropna(subset=[value_col])
    d = d.assign(date=month_start(d[date_col], **mw))
    return d.groupby("date")[value_col].mean().rename(value_col)


def make_model(cps: float = 0.5) -> Any:
    """Make model for this named experimental recipe.

    Notebook context is explicit keyword-only input: none.
    Returns: Any.
    This recipe is diagnostic-only; oracle regressors/holdout tuning are not deployable selection."""
    from prophet import Prophet

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=cps,
        interval_width=0.95,
    )
    m.add_regressor("brent")
    m.add_regressor("usd_index")
    return m


def eval_config(
    cps: float, log_y: bool, *, test: pd.DataFrame, train: pd.DataFrame
) -> dict[str, Any]:
    """Eval config for this named experimental recipe.

    Notebook context is explicit keyword-only input: test, train.
    Returns: dict[str, Any].
    This recipe is diagnostic-only; oracle regressors/holdout tuning are not deployable selection."""
    t = train.copy()
    if log_y:
        t["y"] = np.log(t["y"])
    m = make_model(cps).fit(t)
    pred = m.predict(test[["ds", "brent", "usd_index"]])["yhat"].values
    if log_y:
        pred = np.exp(pred)
    err = pred - test["y"].values
    return {
        "cps": cps,
        "log_y": log_y,
        "MAPE_%": np.mean(np.abs(err / test["y"])) * 100,
        "RMSE_$": np.sqrt(np.mean(err**2)),
    }


def forecast_regressor(
    series: pd.Series, *, future_dates: pd.DatetimeIndex
) -> np.ndarray:
    """Forecast regressor for this named experimental recipe.

    Notebook context is explicit keyword-only input: future_dates.
    Returns: np.ndarray.
    This recipe is diagnostic-only; oracle regressors/holdout tuning are not deployable selection."""
    from prophet import Prophet

    d = series.dropna()
    aux = Prophet(
        yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False
    )
    aux.fit(pd.DataFrame({"ds": d.index, "y": d.values}))
    return aux.predict(pd.DataFrame({"ds": future_dates}))["yhat"].values
