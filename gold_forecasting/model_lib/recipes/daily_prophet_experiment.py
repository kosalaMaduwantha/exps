"""Named diagnostic recipe extracted from gold_daily_prophet.ipynb.

Original algorithms are preserved with explicit context rather than notebook globals.
These experiments may use realized future regressors, full-series screening, or
holdout-driven tuning. They must not publish deployable accuracy/selection claims.
Any annotations mark heterogeneous research/backend results or raw source values.
Use validated_daily for API deployments. No work runs at import time.
"""

from __future__ import annotations
from typing import Any
from datetime import datetime, date
from pathlib import Path
import re
import numpy as np
import pandas as pd
from model_lib.evaluation.diagnostics import error_metrics


def parse_any_date(value: Any) -> Any:
    """Parse any date for this named experimental recipe.

    Notebook context is explicit keyword-only input: none.
    Returns: Any.
    This recipe is diagnostic-only; oracle regressors/holdout tuning are not deployable selection."""
    if pd.isna(value):
        return pd.NaT
    if isinstance(value, (pd.Timestamp, datetime, date, np.datetime64)):
        return pd.Timestamp(value).normalize()
    if isinstance(value, (int, float, np.integer, np.floating)) and (
        not isinstance(value, bool)
    ):
        x = float(value)
        if 10000000 <= x <= 99999999:
            try:
                return pd.Timestamp(
                    datetime.strptime(str(int(x)), "%Y%m%d")
                ).normalize()
            except Exception:
                pass
        if 20000 <= x <= 60000:
            return (
                pd.Timestamp("1899-12-30") + pd.to_timedelta(x, unit="D")
            ).normalize()
    s = str(value).strip()
    if not s or s.lower() in {"nan", "nat", "none"}:
        return pd.NaT
    s = re.sub("^(\\d{1,2})\\.(\\d{1,2})\\.(\\d{4})\\d$", "\\1.\\2.\\3", s)
    if "." in s:
        for fmt in ("%d.%m.%Y", "%d.%m.%y"):
            try:
                return pd.to_datetime(s, format=fmt).normalize()
            except Exception:
                pass
    if "/" in s:
        for fmt in ("%m/%d/%Y", "%d/%m/%Y"):
            try:
                return pd.to_datetime(s, format=fmt).normalize()
            except Exception:
                pass
    try:
        return pd.to_datetime(s, format="ISO8601").normalize()
    except Exception:
        pass
    parsed = pd.to_datetime(s, errors="coerce", dayfirst=True)
    return parsed.normalize() if pd.notna(parsed) else pd.NaT


def load_gold_clean(
    *, DATA: Path, END: Any, START: Any
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load gold clean for this named experimental recipe.

    Notebook context is explicit keyword-only input: DATA, END, START.
    Returns: tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame].
    This recipe is diagnostic-only; oracle regressors/holdout tuning are not deployable selection."""
    raw = pd.read_excel(DATA / "gold_data_new.xlsx", header=None)
    body = raw.iloc[3:].copy()
    body.columns = [
        "row_date",
        "dates",
        "pabc_24",
        "pabc_22",
        "local_24",
        "local_24_biscuit",
        "local_22",
        "global_24_biscuit",
        "global_22",
        "usd_fx",
        "kitco_usd",
        "max_ltv",
        "last_updated",
        "gap",
        "x",
        "y",
    ]
    body = body.drop(columns=["row_date"])
    body["ds"] = body["dates"].map(parse_any_date)
    body["y"] = pd.to_numeric(body["kitco_usd"], errors="coerce")
    dates_ok = body["ds"].notna() & (body["ds"] >= START) & (body["ds"] <= END)
    dropped_dates = body.loc[body["y"].notna() & ~dates_ok, ["dates", "y"]].copy()
    dropped_dates = dropped_dates.rename(
        columns={"dates": "raw_dates", "y": "kitco_usd"}
    )
    dropped_dates["reason"] = "Dates missing or outside valid range"
    filtered = body.loc[dates_ok & body["y"].notna()].copy()
    weekend_rows = int((filtered["ds"].dt.day_of_week >= 5).sum())
    cand = filtered[filtered["ds"].dt.day_of_week < 5].copy()
    counts = cand.groupby("ds").size()
    duplicate_groups = int((counts > 1).sum())
    conflicting_duplicate_groups = int((cand.groupby("ds")["y"].nunique() > 1).sum())
    duplicate_rows_removed = int((counts - 1).clip(lower=0).sum())
    selected_rows = []
    audit_rows = []
    for ds, group in cand.groupby("ds", sort=True):
        if len(group) == 1:
            selected_rows.append({"ds": ds, "y": group["y"].iloc[0]})
            continue
        other = cand.loc[cand["ds"] != ds]
        window = other[
            (other["ds"] >= ds - pd.Timedelta(days=10))
            & (other["ds"] <= ds + pd.Timedelta(days=10))
        ]
        reference = (
            window["y"].median() if window["ds"].nunique() >= 3 else group["y"].median()
        )
        scored = group.assign(score=(group["y"] / reference - 1).abs())
        chosen = scored.sort_values(["score", "y"]).iloc[0]
        rejected = scored.drop(index=chosen.name).sort_values("score")
        selected_rows.append({"ds": ds, "y": chosen["y"]})
        audit_rows.append(
            {
                "ds": ds,
                "n_candidates": len(group),
                "n_distinct_values": group["y"].nunique(),
                "candidate_values": "; ".join(
                    (f"{v:.2f}" for v in sorted(group["y"].unique()))
                ),
                "reference_median": reference,
                "selected_y": chosen["y"],
                "selected_abs_pct": chosen["score"] * 100,
                "rejected_abs_pct": "; ".join(
                    (f"{v * 100:.2f}" for v in rejected["score"])
                ),
            }
        )
    resolved = pd.DataFrame(selected_rows).sort_values("ds").reset_index(drop=True)
    neighbourhood = (
        resolved.set_index("ds")["y"]
        .rolling("21D", center=True, min_periods=3)
        .median()
    )
    resolved["neigh"] = neighbourhood.values
    resolved["neigh_ratio"] = resolved["y"] / resolved["neigh"]
    outlier_mask = (resolved["neigh_ratio"] < 0.85) | (resolved["neigh_ratio"] > 1.15)
    outliers = resolved.loc[outlier_mask, ["ds", "y", "neigh", "neigh_ratio"]].copy()
    resolved.loc[outlier_mask, "y"] = np.nan
    duplicate_audit = pd.DataFrame(audit_rows).sort_values("ds").reset_index(drop=True)
    final = resolved.dropna(subset=["y"])[["ds", "y"]].reset_index(drop=True)
    bdays = pd.bdate_range(final["ds"].min(), final["ds"].max())
    qa = {
        "raw_body_rows": len(body),
        "rows_with_usable_date": int(dates_ok.sum()),
        "rows_in_range_with_target": len(filtered),
        "target_rows_dropped_for_dates": len(dropped_dates),
        "weekend_rows_removed": weekend_rows,
        "duplicate_groups": duplicate_groups,
        "conflicting_duplicate_groups": conflicting_duplicate_groups,
        "duplicate_rows_removed": duplicate_rows_removed,
        "outliers_flagged": int(outlier_mask.sum()),
        "final_observations": len(final),
        "final_start": final["ds"].min(),
        "final_end": final["ds"].max(),
        "missing_business_days": int(len(bdays) - len(final)),
    }
    return (final, qa, outliers, duplicate_audit, dropped_dates)


def load_csv_series(
    filename: str,
    date_col: str,
    value_col: str,
    date_fmt: Any = None,
    na_values: Any = None,
    *,
    DATA: Path,
) -> pd.Series:
    """Load csv series for this named experimental recipe.

    Notebook context is explicit keyword-only input: DATA.
    Returns: pd.Series.
    This recipe is diagnostic-only; oracle regressors/holdout tuning are not deployable selection."""
    df = pd.read_csv(DATA / filename, na_values=na_values)
    if date_fmt is None:
        dates = pd.to_datetime(df[date_col], errors="coerce")
    else:
        dates = pd.to_datetime(df[date_col], format=date_fmt, errors="coerce")
    values = pd.to_numeric(
        df[value_col].astype(str).str.replace(",", "", regex=False), errors="coerce"
    )
    series = pd.Series(values.values, index=dates).dropna().sort_index()
    return series[~series.index.duplicated(keep="last")]


def load_gpr(*, DATA: Path) -> pd.DataFrame:
    """Load gpr for this named experimental recipe.

    Notebook context is explicit keyword-only input: DATA.
    Returns: pd.DataFrame.
    This recipe is diagnostic-only; oracle regressors/holdout tuning are not deployable selection."""
    df = pd.read_excel(DATA / "data_gpr_daily_recent.xls", engine="xlrd")
    df["ds"] = pd.to_datetime(df["DAY"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["ds"]).set_index("ds").sort_index()
    mapping = {
        "gpr": "GPRD",
        "gpr_act": "GPRD_ACT",
        "gpr_threat": "GPRD_THREAT",
        "gpr_ma7": "GPRD_MA7",
        "gpr_ma30": "GPRD_MA30",
    }
    return pd.DataFrame(
        {name: pd.to_numeric(df[col], errors="coerce") for name, col in mapping.items()}
    )


def load_cpi(*, DATA: Path) -> pd.Series:
    """Load cpi for this named experimental recipe.

    Notebook context is explicit keyword-only input: DATA.
    Returns: pd.Series.
    This recipe is diagnostic-only; oracle regressors/holdout tuning are not deployable selection."""
    raw = pd.read_excel(
        DATA / "SeriesReport-20261028041515_7e01eb-use.xlsx", header=None
    )
    header_row = raw.index[raw[0].astype(str).str.strip().eq("Series ID")][0]
    cpi = raw.iloc[header_row + 1 :].copy()
    cpi.columns = ["series", "year", "period", "value"]
    cpi = cpi.dropna(subset=["year"])
    cpi["date"] = pd.to_datetime(
        cpi["year"].astype(int).astype(str)
        + cpi["period"].astype(str).str.replace("M", "-"),
        format="%Y-%m",
        errors="coerce",
    )
    cpi["value"] = pd.to_numeric(cpi["value"], errors="coerce")
    cpi = cpi.dropna(subset=["date", "value"]).set_index("date").sort_index()["value"]
    return cpi


def monthly_asof(
    series: pd.Series, release_days: Any, target_index: pd.DatetimeIndex
) -> pd.Series:
    """Monthly asof for this named experimental recipe.

    Notebook context is explicit keyword-only input: none.
    Returns: pd.Series.
    This recipe is diagnostic-only; oracle regressors/holdout tuning are not deployable selection."""
    s = series.dropna().copy()
    s.index = s.index.to_period("M")
    release = s.index.to_timestamp(how="end").normalize() + pd.Timedelta(
        days=release_days
    )
    frame = pd.DataFrame({"release": release, "value": s.values}).sort_values("release")
    query = pd.DataFrame({"ds": target_index}).sort_values("ds")
    merged = pd.merge_asof(
        query, frame, left_on="ds", right_on="release", direction="backward"
    )
    return pd.Series(merged["value"].values, index=merged["ds"].values)


def add_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add features for this named experimental recipe.

    Notebook context is explicit keyword-only input: none.
    Returns: pd.DataFrame.
    This recipe is diagnostic-only; oracle regressors/holdout tuning are not deployable selection."""
    p = panel.copy()
    p["gold_log"] = np.log(p["gold"].where(p["gold"] > 0))
    p["gold_ret_1d"] = p["gold_log"].diff(1)
    p["gold_ret_5d"] = p["gold_log"].diff(5)
    p["gold_ret_20d"] = p["gold_log"].diff(20)
    p["gold_vol_20d"] = p["gold_ret_1d"].rolling(20).std()
    for column in ["sp500", "brent", "dxy"]:
        p[column + "_log"] = np.log(p[column].where(p[column] > 0))
        for horizon in (1, 5, 20, 60, 252):
            p[f"{column}_ret_{horizon}d"] = p[column + "_log"].diff(horizon)
    for horizon in (1, 5, 20, 60, 252):
        p[f"wti_chg_{horizon}d"] = p["wti"].diff(horizon)
    for column in ["gpr", "gpr_ma7", "gpr_ma30"]:
        for horizon in (1, 5, 20, 60, 252):
            p[f"{column}_ret_{horizon}d"] = p[column].diff(horizon)
    for column in ["ust10y", "fed_funds", "cpi_yoy", "real_yield"]:
        for horizon in (1, 5, 20, 60, 252):
            p[f"{column}_chg_{horizon}d"] = p[column].diff(horizon)
    for column in ["gpr", "gpr_ma7", "gpr_threat", "brent", "dxy", "ust10y", "sp500"]:
        p[f"{column}_ma20"] = p[column].rolling(20).mean()
        p[f"{column}_sd20"] = p[column].rolling(20).std()
        p[f"{column}_z20"] = (p[column] - p[f"{column}_ma20"]) / p[f"{column}_sd20"]
    return p


def ccf_table(
    df: pd.DataFrame, target: str, features: list[str], max_lag: int = 10
) -> pd.DataFrame:
    """Ccf table for this named experimental recipe.

    Notebook context is explicit keyword-only input: none.
    Returns: pd.DataFrame.
    This recipe is diagnostic-only; oracle regressors/holdout tuning are not deployable selection."""
    rows = []
    for feature in features:
        for lag in range(-max_lag, max_lag + 1):
            pair = pd.concat([df[target], df[feature].shift(lag)], axis=1)
            pair = pair.replace([np.inf, -np.inf], np.nan).dropna()
            if len(pair) > 30:
                r = pair.iloc[:, 0].corr(pair.iloc[:, 1])
                if np.isfinite(r):
                    rows.append(
                        {"feature": feature, "lag": lag, "r": r, "n": len(pair)}
                    )
    return pd.DataFrame(rows)


def pair_corr(left: pd.Series, right: pd.Series) -> tuple[float, int]:
    """Pair corr for this named experimental recipe.

    Notebook context is explicit keyword-only input: none.
    Returns: tuple[float, int].
    This recipe is diagnostic-only; oracle regressors/holdout tuning are not deployable selection."""
    pair = pd.concat([left, right], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    return (float(pair.iloc[:, 0].corr(pair.iloc[:, 1])), len(pair))


def fit_and_evaluate(
    regressors: list[str],
    cps: float,
    holdout_start: Any,
    horizon_months: int = 24,
    *,
    model_df: pd.DataFrame,
) -> dict[str, Any]:
    """Fit and evaluate for this named experimental recipe.

    Notebook context is explicit keyword-only input: model_df.
    Returns: dict[str, Any].
    This recipe is diagnostic-only; oracle regressors/holdout tuning are not deployable selection."""
    from prophet import Prophet

    train = model_df[model_df["ds"] < holdout_start].copy()
    end = holdout_start + pd.DateOffset(months=horizon_months)
    test = model_df[(model_df["ds"] >= holdout_start) & (model_df["ds"] < end)].copy()
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=cps,
        interval_width=0.95,
        seasonality_mode="multiplicative",
    )
    for regressor in regressors:
        model.add_regressor(regressor)
    train_log = train.copy()
    train_log["y"] = np.log(train_log["gold"])
    model.fit(train_log)
    train_fit = model.predict(train_log[["ds"] + regressors])
    last_fitted = np.exp(train_fit["yhat"].iloc[-1])
    last_actual = train["gold"].iloc[-1]
    anchor_ratio = last_actual / last_fitted
    forecast = model.predict(test[["ds"] + regressors])
    raw_prediction = np.exp(forecast["yhat"].values)
    anchored_prediction = raw_prediction * anchor_ratio
    test_out = test[["ds", "gold"]].copy()
    test_out["pred_raw"] = raw_prediction
    test_out["pred"] = anchored_prediction
    test_out["resid"] = test_out["pred"] - test_out["gold"]
    result = {
        "test": test_out,
        "model": model,
        "anchor_ratio": anchor_ratio,
        "train_n": len(train),
        "test_n": len(test),
    }
    result.update(error_metrics(test_out["gold"], test_out["pred"]))
    raw_metrics = error_metrics(test_out["gold"], test_out["pred_raw"])
    result["MAPE_raw"] = raw_metrics["MAPE"]
    result["RMSE_raw"] = raw_metrics["RMSE"]
    return result


def anchored_prophet(
    series: pd.Series, freq: str, periods: int, yearly: bool, cps: float
) -> pd.DataFrame:
    """Anchored prophet for this named experimental recipe.

    Notebook context is explicit keyword-only input: none.
    Returns: pd.DataFrame.
    This recipe is diagnostic-only; oracle regressors/holdout tuning are not deployable selection."""
    from prophet import Prophet

    s = series.dropna().sort_index()
    model = Prophet(
        yearly_seasonality=yearly,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=cps,
        interval_width=0.95,
    )
    model.fit(pd.DataFrame({"ds": s.index, "y": s.values}))
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)[["ds", "yhat"]]
    last_fit = model.predict(pd.DataFrame({"ds": [s.index.max()]}))["yhat"].iloc[-1]
    forecast["yhat"] = forecast["yhat"] + (s.iloc[-1] - last_fit)
    return forecast


def monthly_forecast(
    series: pd.Series, periods: int = 12, yearly: bool = True
) -> pd.DataFrame:
    """Monthly forecast for this named experimental recipe.

    Notebook context is explicit keyword-only input: none.
    Returns: pd.DataFrame.
    This recipe is diagnostic-only; oracle regressors/holdout tuning are not deployable selection."""
    from prophet import Prophet

    s = series.copy()
    s.index = s.index.to_period("M").to_timestamp("M")
    s = s[~s.index.duplicated(keep="last")].sort_index()
    model = Prophet(
        yearly_seasonality=yearly,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        interval_width=0.95,
    )
    model.fit(pd.DataFrame({"ds": s.index, "y": s.values}))
    future = model.make_future_dataframe(periods=periods, freq="ME")
    forecast = model.predict(future)[["ds", "yhat"]]
    last_fit = model.predict(pd.DataFrame({"ds": [s.index.max()]}))["yhat"].iloc[-1]
    forecast["yhat"] = forecast["yhat"] + (s.iloc[-1] - last_fit)
    return forecast


def monthly_to_daily(
    history: Any, forecast: Any, release_lag: Any, target_index: pd.DatetimeIndex
) -> pd.Series:
    """Monthly to daily for this named experimental recipe.

    Notebook context is explicit keyword-only input: none.
    Returns: pd.Series.
    This recipe is diagnostic-only; oracle regressors/holdout tuning are not deployable selection."""
    hist = pd.DataFrame({"ds": history.index, "value": history.values})
    hist["release"] = hist["ds"] + pd.Timedelta(days=release_lag)
    fut = forecast[forecast["ds"] > history.index.max()][["ds", "yhat"]].copy()
    fut = fut.rename(columns={"yhat": "value"})
    fut["release"] = fut["ds"] + pd.Timedelta(days=release_lag)
    frame = pd.concat([hist[["release", "value"]], fut[["release", "value"]]])
    frame = frame.sort_values("release")
    query = pd.DataFrame({"ds": target_index})
    merged = pd.merge_asof(
        query, frame, left_on="ds", right_on="release", direction="backward"
    )
    return pd.Series(merged["value"].values, index=merged["ds"].values)
