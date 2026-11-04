"""Named diagnostic recipe extracted from gold_daily_xgboost.ipynb.

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
from model_lib.evaluation.diagnostics import error_metrics, PointErrors


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Rsi for this named experimental recipe.

    Notebook context is explicit keyword-only input: none.
    Returns: pd.Series.
    This recipe is diagnostic-only; oracle regressors/holdout tuning are not deployable selection."""
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def target_calendar(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Target calendar for this named experimental recipe.

    Notebook context is explicit keyword-only input: none.
    Returns: pd.DataFrame.
    This recipe is diagnostic-only; oracle regressors/holdout tuning are not deployable selection."""
    dates = pd.DatetimeIndex(dates)
    frame = pd.DataFrame(index=dates)
    frame["t_dow"] = dates.dayofweek.astype(float)
    frame["t_month_sin"] = np.sin(2 * np.pi * (dates.month - 1) / 12)
    frame["t_month_cos"] = np.cos(2 * np.pi * (dates.month - 1) / 12)
    frame["t_is_month_end"] = dates.is_month_end.astype(float)
    frame["t_is_quarter_end"] = dates.is_quarter_end.astype(float)
    return frame


def origin_calendar(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Origin calendar for this named experimental recipe.

    Notebook context is explicit keyword-only input: none.
    Returns: pd.DataFrame.
    This recipe is diagnostic-only; oracle regressors/holdout tuning are not deployable selection."""
    dates = pd.DatetimeIndex(dates)
    frame = pd.DataFrame(index=dates)
    frame["dow"] = dates.dayofweek.astype(float)
    frame["month_sin"] = np.sin(2 * np.pi * (dates.month - 1) / 12)
    frame["month_cos"] = np.cos(2 * np.pi * (dates.month - 1) / 12)
    frame["is_month_end"] = dates.is_month_end.astype(float)
    frame["is_quarter_end"] = dates.is_quarter_end.astype(float)
    return frame


def build_features(gold_series: pd.Series, external: pd.DataFrame) -> pd.DataFrame:
    """Build features for this named experimental recipe.

    Notebook context is explicit keyword-only input: none.
    Returns: pd.DataFrame.
    This recipe is diagnostic-only; oracle regressors/holdout tuning are not deployable selection."""
    gold = gold_series.astype(float)
    gold_ff = gold.ffill()
    log_gold = np.log(gold_ff.where(gold_ff > 0))
    frame = pd.DataFrame(index=gold.index)
    for n in (1, 5, 10, 21, 63, 126, 252):
        frame[f"gold_ret_{n}"] = log_gold.diff(n)
    for n in (5, 21, 63):
        frame[f"gold_vol_{n}"] = frame["gold_ret_1"].rolling(n).std()
    for n in (10, 20, 50, 200):
        frame[f"gold_ma_dist_{n}"] = log_gold - np.log(gold_ff.rolling(n).mean())
    frame["gold_rsi_14"] = rsi(gold_ff, 14)
    ema12 = gold_ff.ewm(span=12, adjust=False).mean()
    ema26 = gold_ff.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    frame["gold_macd_over_px"] = macd / gold_ff
    frame["gold_macd_hist_over_px"] = (macd - signal) / gold_ff
    ma20 = gold_ff.rolling(20).mean()
    sd20 = gold_ff.rolling(20).std()
    frame["gold_bollinger_z_20"] = (gold_ff - ma20) / sd20.replace(0, np.nan)
    frame["gold_drawdown_252"] = log_gold - np.log(gold_ff.rolling(252).max())
    frame["gold_skew_63"] = frame["gold_ret_1"].rolling(63).skew()
    frame["gold_kurt_63"] = frame["gold_ret_1"].rolling(63).kurt()
    frame["gold_ret_z_20"] = frame["gold_ret_1"] / frame["gold_vol_21"].replace(
        0, np.nan
    )
    positions = pd.Series(np.arange(len(gold)), index=gold.index)
    last_obs_pos = positions.where(gold.notna()).ffill()
    frame["days_since_obs"] = (positions - last_obs_pos).fillna(999).astype(float)
    ext = external.reindex(gold.index)
    ext_lag = ext.shift(1)
    for column in ["ust10y", "dxy", "fed_funds", "cpi_yoy", "real_yield"]:
        frame[f"{column}_level"] = ext_lag[column]
        for n in (1, 5, 21, 63):
            frame[f"{column}_chg_{n}"] = ext_lag[column].diff(n)
        frame[f"{column}_z252"] = (
            ext_lag[column] - ext_lag[column].rolling(252).mean()
        ) / ext_lag[column].rolling(252).std().replace(0, np.nan)
    sp500_log = np.log(ext_lag["sp500"].where(ext_lag["sp500"] > 0))
    for n in (1, 5, 21, 63):
        frame[f"sp500_ret_{n}"] = sp500_log.diff(n)
    frame["sp500_vol_21"] = sp500_log.diff(1).rolling(21).std()
    frame["sp500_z252"] = (
        ext_lag["sp500"] - ext_lag["sp500"].rolling(252).mean()
    ) / ext_lag["sp500"].rolling(252).std().replace(0, np.nan)
    frame["gpr_ma7_level"] = ext_lag["gpr_ma7"]
    for n in (1, 5, 21, 63):
        frame[f"gpr_ma7_chg_{n}"] = ext_lag["gpr_ma7"].diff(n)
    frame["gpr_ma7_z252"] = (
        ext_lag["gpr_ma7"] - ext_lag["gpr_ma7"].rolling(252).mean()
    ) / ext_lag["gpr_ma7"].rolling(252).std().replace(0, np.nan)
    brent_log = np.log(ext_lag["brent"].where(ext_lag["brent"] > 0))
    for n in (1, 5, 21, 63):
        frame[f"brent_ret_{n}"] = brent_log.diff(n)
    frame["brent_vol_21"] = brent_log.diff(1).rolling(21).std()
    for n in (1, 5, 21, 63):
        frame[f"wti_chg_{n}"] = ext_lag["wti"].diff(n)
    frame["wti_vol_21"] = ext_lag["wti"].diff(1).rolling(21).std()
    frame["gold_minus_dxy_21"] = frame["gold_ret_21"] - frame["dxy_chg_21"]
    frame["gold_minus_sp500_21"] = frame["gold_ret_21"] - frame["sp500_ret_21"]
    frame["gold_minus_real_yield_21"] = (
        frame["gold_ret_21"] - frame["real_yield_chg_21"]
    )
    return frame


def build_direct_dataset(
    feature_names: list[str],
    hmax: int,
    cutoff: Any = None,
    stride: int = 3,
    *,
    DRIFT: pd.Series,
    DYN_HIST: pd.DataFrame,
    FUTURE_CORE: list[str],
    F_hist: pd.DataFrame,
    TARGET_CAL: list[str],
    bdays: pd.DatetimeIndex,
    obs: pd.Series,
    target: pd.Series,
) -> pd.DataFrame:
    """Build direct dataset for this named experimental recipe.

    Notebook context is explicit keyword-only input: DRIFT, DYN_HIST, FUTURE_CORE, F_hist, TARGET_CAL, bdays, obs, target.
    Returns: pd.DataFrame.
    This recipe is diagnostic-only; oracle regressors/holdout tuning are not deployable selection."""
    origins = np.where(obs.to_numpy())[0]
    if cutoff is not None:
        origins = origins[bdays[origins] < cutoff]
    if stride > 1 and len(origins) > 0:
        selected = origins[::stride]
        if selected[-1] != origins[-1]:
            selected = np.append(selected, origins[-1])
        origins = selected
    dynamic = [c for c in feature_names if c in FUTURE_CORE or c in TARGET_CAL]
    static = [c for c in feature_names if c not in dynamic]
    X0 = F_hist[static].to_numpy(dtype=np.float32)
    D0 = DYN_HIST[dynamic].to_numpy(dtype=np.float32)
    gold = target.to_numpy(dtype=float)
    drift = DRIFT.to_numpy(dtype=float)
    rows = []
    for h in range(1, hmax + 1):
        target_positions = origins + h
        valid = target_positions < len(bdays)
        safe_positions = np.clip(target_positions, 0, len(bdays) - 1)
        if cutoff is not None:
            valid &= bdays[safe_positions] < cutoff
        valid &= np.isfinite(gold[safe_positions]) & np.isfinite(gold[origins])
        if not valid.any():
            continue
        origin_pos = origins[valid]
        target_pos = target_positions[valid]
        horizon_col = np.full((len(origin_pos), 1), h, dtype=np.float32)
        X = np.column_stack([horizon_col, X0[origin_pos], D0[target_pos]])
        frame = pd.DataFrame(X, columns=["horizon"] + static + dynamic)
        frame["origin_ds"] = bdays[origin_pos]
        frame["target_ds"] = bdays[target_pos]
        frame["h"] = h
        frame["y"] = np.log(gold[target_pos] / gold[origin_pos]) / h - drift[origin_pos]
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def split_feature_names(
    feature_names: list[str], *, FUTURE_CORE: list[str], TARGET_CAL: list[str]
) -> tuple[list[str], list[str]]:
    """Split feature names for this named experimental recipe.

    Notebook context is explicit keyword-only input: FUTURE_CORE, TARGET_CAL.
    Returns: tuple[list[str], list[str]].
    This recipe is diagnostic-only; oracle regressors/holdout tuning are not deployable selection."""
    dynamic = [c for c in feature_names if c in FUTURE_CORE or c in TARGET_CAL]
    static = [c for c in feature_names if c not in dynamic]
    return (static, dynamic)


def predict_direct(
    model: Any,
    feature_names: list[str],
    origin_date: Any,
    target_dates: pd.DatetimeIndex,
    future: bool = False,
    iteration_range: Any = None,
    *,
    DRIFT: pd.Series,
    DYN_FUT: pd.DataFrame,
    DYN_HIST: pd.DataFrame,
    FUTURE_CORE: list[str],
    F_hist: pd.DataFrame,
    TARGET_CAL: list[str],
    bdays: pd.DatetimeIndex,
    target: pd.Series,
) -> pd.DataFrame:
    """Predict direct for this named experimental recipe.

    Notebook context is explicit keyword-only input: DRIFT, DYN_FUT, DYN_HIST, FUTURE_CORE, F_hist, TARGET_CAL, bdays, target.
    Returns: pd.DataFrame.
    This recipe is diagnostic-only; oracle regressors/holdout tuning are not deployable selection."""
    static, dynamic = split_feature_names(
        feature_names, FUTURE_CORE=FUTURE_CORE, TARGET_CAL=TARGET_CAL
    )
    origin_date = pd.Timestamp(origin_date)
    target_dates = pd.DatetimeIndex(target_dates)
    origin_pos = bdays.get_loc(origin_date)
    if future:
        h = np.arange(1, len(target_dates) + 1)
        D = DYN_FUT.loc[target_dates]
    else:
        target_pos = bdays.get_indexer(target_dates)
        if (target_pos < 0).any():
            raise ValueError("Target dates not present in the business-day index")
        h = target_pos - origin_pos
        D = DYN_HIST.loc[target_dates]
    X_static = np.repeat(
        F_hist.loc[[origin_date], static].to_numpy(dtype=np.float32),
        len(target_dates),
        axis=0,
    )
    X = np.column_stack(
        [h.astype(np.float32)[:, None], X_static, D[dynamic].to_numpy(dtype=np.float32)]
    )
    kwargs = {"iteration_range": iteration_range} if iteration_range is not None else {}
    prediction = model.predict(X, **kwargs)
    origin_price = float(target.loc[origin_date])
    drift = float(DRIFT.loc[origin_date])
    return pd.DataFrame(
        {
            "ds": target_dates,
            "pred_logret": prediction + drift,
            "pred": origin_price * np.exp((prediction + drift) * h),
            "h": h,
        }
    )


def fit_direct_model(
    feature_names: list[str],
    cutoff: Any,
    hmax: int,
    params: dict[str, Any],
    stride: int = 3,
    *,
    DRIFT: pd.Series,
    DYN_HIST: pd.DataFrame,
    FUTURE_CORE: list[str],
    F_hist: pd.DataFrame,
    TARGET_CAL: list[str],
    bdays: pd.DatetimeIndex,
    obs: pd.Series,
    target: pd.Series,
) -> tuple[Any, pd.DataFrame, list[str]]:
    """Fit direct model for this named experimental recipe.

    Notebook context is explicit keyword-only input: DRIFT, DYN_HIST, FUTURE_CORE, F_hist, TARGET_CAL, bdays, obs, target.
    Returns: tuple[Any, pd.DataFrame, list[str]].
    This recipe is diagnostic-only; oracle regressors/holdout tuning are not deployable selection."""
    from xgboost import XGBRegressor

    train = build_direct_dataset(
        feature_names,
        hmax=hmax,
        cutoff=cutoff,
        stride=stride,
        DRIFT=DRIFT,
        DYN_HIST=DYN_HIST,
        FUTURE_CORE=FUTURE_CORE,
        F_hist=F_hist,
        TARGET_CAL=TARGET_CAL,
        bdays=bdays,
        obs=obs,
        target=target,
    )
    static, dynamic = split_feature_names(
        feature_names, FUTURE_CORE=FUTURE_CORE, TARGET_CAL=TARGET_CAL
    )
    columns = ["horizon"] + static + dynamic
    model = XGBRegressor(**params)
    model.fit(train[columns], train["y"])
    return (model, train, columns)


def eval_direct(
    feature_names: list[str],
    cutoff: Any,
    months: int,
    hmax: int,
    params: dict[str, Any],
    stride: int = 3,
    *,
    DRIFT: pd.Series,
    DYN_FUT: pd.DataFrame,
    DYN_HIST: pd.DataFrame,
    FUTURE_CORE: list[str],
    F_hist: pd.DataFrame,
    TARGET_CAL: list[str],
    bdays: pd.DatetimeIndex,
    obs: pd.Series,
    target: pd.Series,
) -> dict[str, Any]:
    """Eval direct for this named experimental recipe.

    Notebook context is explicit keyword-only input: DRIFT, DYN_FUT, DYN_HIST, FUTURE_CORE, F_hist, TARGET_CAL, bdays, obs, target.
    Returns: dict[str, Any].
    This recipe is diagnostic-only; oracle regressors/holdout tuning are not deployable selection."""
    model, train, columns = fit_direct_model(
        feature_names,
        cutoff,
        hmax,
        params,
        stride,
        DRIFT=DRIFT,
        DYN_HIST=DYN_HIST,
        FUTURE_CORE=FUTURE_CORE,
        F_hist=F_hist,
        TARGET_CAL=TARGET_CAL,
        bdays=bdays,
        obs=obs,
        target=target,
    )
    origin_date = target.loc[target.index < cutoff].dropna().index.max()
    end_date = cutoff + pd.DateOffset(months=months)
    test = target[(target.index >= cutoff) & (target.index < end_date)].dropna()
    predicted = predict_direct(
        model,
        feature_names,
        origin_date,
        test.index,
        DRIFT=DRIFT,
        DYN_FUT=DYN_FUT,
        DYN_HIST=DYN_HIST,
        FUTURE_CORE=FUTURE_CORE,
        F_hist=F_hist,
        TARGET_CAL=TARGET_CAL,
        bdays=bdays,
        target=target,
    )
    test_frame = pd.DataFrame({"ds": test.index, "actual": test.values}).merge(
        predicted, on="ds", how="left"
    )
    metrics = error_metrics(test_frame["actual"], test_frame["pred"])
    return {
        "metrics": metrics,
        "test": test_frame,
        "model": model,
        "train": train,
        "columns": columns,
        "origin_date": origin_date,
        "train_n": len(train),
        "test_n": len(test_frame),
    }


def eval_random_walk(cutoff: Any, months: int, *, target: pd.Series) -> PointErrors:
    """Eval random walk for this named experimental recipe.

    Notebook context is explicit keyword-only input: target.
    Returns: dict[str, Any].
    This recipe is diagnostic-only; oracle regressors/holdout tuning are not deployable selection."""
    origin_date = target.loc[target.index < cutoff].dropna().index.max()
    end_date = cutoff + pd.DateOffset(months=months)
    test = target[(target.index >= cutoff) & (target.index < end_date)].dropna()
    predicted = np.full(len(test), float(target.loc[origin_date]))
    return error_metrics(test.to_numpy(), predicted)


def build_recursive_dataset(
    feature_names: list[str],
    cutoff: Any,
    stride: int = 1,
    *,
    DRIFT: pd.Series,
    DYN_HIST: pd.DataFrame,
    FUTURE_CORE: list[str],
    F_hist: pd.DataFrame,
    TARGET_CAL: list[str],
    bdays: pd.DatetimeIndex,
    obs: pd.Series,
    target: pd.Series,
) -> pd.DataFrame:
    """Build recursive dataset for this named experimental recipe.

    Notebook context is explicit keyword-only input: DRIFT, DYN_HIST, FUTURE_CORE, F_hist, TARGET_CAL, bdays, obs, target.
    Returns: pd.DataFrame.
    This recipe is diagnostic-only; oracle regressors/holdout tuning are not deployable selection."""
    origins = np.where(obs.to_numpy())[0]
    origins = origins[origins + 1 < len(bdays)]
    origins = origins[bdays[origins] < cutoff]
    next_positions = origins + 1
    gold = target.to_numpy(dtype=float)
    valid = np.isfinite(gold[next_positions]) & np.isfinite(gold[origins])
    origins = origins[valid]
    next_positions = next_positions[valid]
    if stride > 1 and len(origins) > 0:
        selected = origins[::stride]
        if selected[-1] != origins[-1]:
            selected = np.append(selected, origins[-1])
        origins = selected
        next_positions = origins + 1
    dynamic = [c for c in feature_names if c in FUTURE_CORE or c in TARGET_CAL]
    static = [c for c in feature_names if c not in dynamic]
    X0 = F_hist[static].to_numpy(dtype=np.float32)
    D0 = DYN_HIST[dynamic].to_numpy(dtype=np.float32)
    horizon_col = np.ones((len(origins), 1), dtype=np.float32)
    X = np.column_stack([horizon_col, X0[origins], D0[next_positions]])
    frame = pd.DataFrame(X, columns=["horizon"] + static + dynamic)
    frame["origin_ds"] = bdays[origins]
    frame["target_ds"] = bdays[next_positions]
    frame["h"] = 1
    frame["y"] = (
        np.log(gold[next_positions] / gold[origins])
        - DRIFT.to_numpy(dtype=float)[origins]
    )
    return frame


def predict_recursive(
    model: Any,
    feature_names: list[str],
    origin_date: Any,
    target_dates: pd.DatetimeIndex,
    final: bool = False,
    *,
    DRIFT_CAP: float,
    DYN_FUT: pd.DataFrame,
    DYN_HIST: pd.DataFrame,
    FUTURE_CORE: list[str],
    TARGET_CAL: list[str],
    ext_all: pd.DataFrame,
    target: pd.Series,
) -> pd.DataFrame:
    """Predict recursive for this named experimental recipe.

    Notebook context is explicit keyword-only input: DRIFT_CAP, DYN_FUT, DYN_HIST, FUTURE_CORE, TARGET_CAL, ext_all, target.
    Returns: pd.DataFrame.
    This recipe is diagnostic-only; oracle regressors/holdout tuning are not deployable selection."""
    static, dynamic = split_feature_names(
        feature_names, FUTURE_CORE=FUTURE_CORE, TARGET_CAL=TARGET_CAL
    )
    origin_date = pd.Timestamp(origin_date)
    target_dates = pd.DatetimeIndex(target_dates)
    gold_path = target.reindex(target.index.union(target_dates)).ffill()
    current_price = float(target.loc[origin_date])
    previous = origin_date
    output: list[dict[str, Any]] = []
    for target_date in target_dates:
        features = build_features(gold_path, ext_all).join(
            origin_calendar(gold_path.index)
        )
        static_row = features.loc[[previous], static].to_numpy(dtype=np.float32)
        dynamic_source = DYN_FUT if final else DYN_HIST
        dynamic_row = dynamic_source.loc[[target_date], dynamic].to_numpy(
            dtype=np.float32
        )
        X = np.column_stack(
            [np.ones((1, 1), dtype=np.float32), static_row, dynamic_row]
        )
        residual = float(model.predict(X)[0])
        drift_series = (np.log(gold_path.ffill()).diff(252) / 252.0).clip(
            -DRIFT_CAP, DRIFT_CAP
        )
        drift = (
            float(drift_series.loc[target_date])
            if pd.notna(drift_series.loc[target_date])
            else 0.0
        )
        current_price = current_price * np.exp(residual + drift)
        gold_path.loc[target_date] = current_price
        output.append(
            {
                "ds": target_date,
                "pred": current_price,
                "pred_logret": residual + drift,
                "h": len(output) + 1,
            }
        )
        previous = target_date
    return pd.DataFrame(output)


def eval_recursive(
    feature_names: list[str],
    cutoff: Any,
    months: int,
    params: dict[str, Any],
    stride: int = 1,
    *,
    DRIFT: pd.Series,
    DRIFT_CAP: float,
    DYN_FUT: pd.DataFrame,
    DYN_HIST: pd.DataFrame,
    FUTURE_CORE: list[str],
    F_hist: pd.DataFrame,
    TARGET_CAL: list[str],
    bdays: pd.DatetimeIndex,
    ext_all: pd.DataFrame,
    obs: pd.Series,
    target: pd.Series,
) -> dict[str, Any]:
    """Eval recursive for this named experimental recipe.

    Notebook context is explicit keyword-only input: DRIFT, DRIFT_CAP, DYN_FUT, DYN_HIST, FUTURE_CORE, F_hist, TARGET_CAL, bdays, ext_all, obs, target.
    Returns: dict[str, Any].
    This recipe is diagnostic-only; oracle regressors/holdout tuning are not deployable selection."""
    from xgboost import XGBRegressor

    train = build_recursive_dataset(
        feature_names,
        cutoff,
        stride=stride,
        DRIFT=DRIFT,
        DYN_HIST=DYN_HIST,
        FUTURE_CORE=FUTURE_CORE,
        F_hist=F_hist,
        TARGET_CAL=TARGET_CAL,
        bdays=bdays,
        obs=obs,
        target=target,
    )
    static, dynamic = split_feature_names(
        feature_names, FUTURE_CORE=FUTURE_CORE, TARGET_CAL=TARGET_CAL
    )
    columns = ["horizon"] + static + dynamic
    model = XGBRegressor(**params)
    model.fit(train[columns], train["y"])
    origin_date = target.loc[target.index < cutoff].dropna().index.max()
    end_date = cutoff + pd.DateOffset(months=months)
    test = target[(target.index >= cutoff) & (target.index < end_date)].dropna()
    target_dates = pd.bdate_range(origin_date + pd.Timedelta(days=1), test.index.max())
    predicted = predict_recursive(
        model,
        feature_names,
        origin_date,
        target_dates,
        final=False,
        DRIFT_CAP=DRIFT_CAP,
        DYN_FUT=DYN_FUT,
        DYN_HIST=DYN_HIST,
        FUTURE_CORE=FUTURE_CORE,
        TARGET_CAL=TARGET_CAL,
        ext_all=ext_all,
        target=target,
    )
    test_frame = pd.DataFrame({"ds": test.index, "actual": test.values}).merge(
        predicted, on="ds", how="left"
    )
    return {
        "metrics": error_metrics(test_frame["actual"], test_frame["pred"]),
        "test": test_frame,
        "train_n": len(train),
        "test_n": len(test_frame),
        "origin_date": origin_date,
    }


def fit_quantile_models(
    feature_names: list[str],
    cutoff: Any,
    hmax: int,
    params: dict[str, Any],
    stride: int,
    *,
    DRIFT: pd.Series,
    DYN_HIST: pd.DataFrame,
    FUTURE_CORE: list[str],
    F_hist: pd.DataFrame,
    TARGET_CAL: list[str],
    bdays: pd.DatetimeIndex,
    obs: pd.Series,
    target: pd.Series,
) -> tuple[dict[float, Any], pd.DataFrame]:
    """Fit quantile models for this named experimental recipe.

    Notebook context is explicit keyword-only input: DRIFT, DYN_HIST, FUTURE_CORE, F_hist, TARGET_CAL, bdays, obs, target.
    Returns: tuple[dict[float, Any], pd.DataFrame].
    This recipe is diagnostic-only; oracle regressors/holdout tuning are not deployable selection."""
    from xgboost import XGBRegressor

    train = build_direct_dataset(
        feature_names,
        hmax=hmax,
        cutoff=cutoff,
        stride=stride,
        DRIFT=DRIFT,
        DYN_HIST=DYN_HIST,
        FUTURE_CORE=FUTURE_CORE,
        F_hist=F_hist,
        TARGET_CAL=TARGET_CAL,
        bdays=bdays,
        obs=obs,
        target=target,
    )
    static, dynamic = split_feature_names(
        feature_names, FUTURE_CORE=FUTURE_CORE, TARGET_CAL=TARGET_CAL
    )
    columns = ["horizon"] + static + dynamic
    models = {}
    for alpha in (0.025, 0.5, 0.975):
        quantile_params = dict(params)
        quantile_params["objective"] = "reg:quantileerror"
        quantile_params["quantile_alpha"] = alpha
        quantile_params["n_estimators"] = min(int(params["n_estimators"]), 300)
        model = XGBRegressor(**quantile_params)
        model.fit(train[columns], train["y"])
        models[alpha] = model
    return (models, train)


def interval_predictions(
    models: dict[float, Any],
    feature_names: list[str],
    origin_date: Any,
    target_dates: pd.DatetimeIndex,
    future: bool = False,
    *,
    DRIFT: pd.Series,
    DYN_FUT: pd.DataFrame,
    DYN_HIST: pd.DataFrame,
    FUTURE_CORE: list[str],
    F_hist: pd.DataFrame,
    TARGET_CAL: list[str],
    bdays: pd.DatetimeIndex,
    target: pd.Series,
) -> pd.DataFrame:
    """Interval predictions for this named experimental recipe.

    Notebook context is explicit keyword-only input: DRIFT, DYN_FUT, DYN_HIST, FUTURE_CORE, F_hist, TARGET_CAL, bdays, target.
    Returns: pd.DataFrame.
    This recipe is diagnostic-only; oracle regressors/holdout tuning are not deployable selection."""
    pieces = []
    for alpha, model in models.items():
        prediction = predict_direct(
            model,
            feature_names,
            origin_date,
            target_dates,
            future=future,
            DRIFT=DRIFT,
            DYN_FUT=DYN_FUT,
            DYN_HIST=DYN_HIST,
            FUTURE_CORE=FUTURE_CORE,
            F_hist=F_hist,
            TARGET_CAL=TARGET_CAL,
            bdays=bdays,
            target=target,
        )
        prediction = prediction[["ds", "pred"]].rename(columns={"pred": f"q{alpha}"})
        pieces.append(prediction)
    frame = pieces[0]
    for piece in pieces[1:]:
        frame = frame.merge(piece, on="ds", how="inner")
    lower = frame[["q0.025", "q0.5", "q0.975"]].min(axis=1)
    upper = frame[["q0.025", "q0.5", "q0.975"]].max(axis=1)
    frame["quantile_lower_95"] = lower
    frame["quantile_upper_95"] = upper
    frame["quantile_median"] = frame["q0.5"]
    return frame
