"""Gold preprocessing / features. Extracted default daily algorithms; see function contracts.

Model-fitting workflows are invoked only by notebooks, never by serving.
"""

from __future__ import annotations


import numpy as np
import pandas as pd
from model_lib.contracts import Config


def make_features(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Create causal candidate features for the model notebooks.

    Only information available at the origin appears in a row: lagged and
    trailing gold statistics, carried external values, their ages/missingness,
    and deterministic calendar attributes. Gold-derived features are computed
    from observed quotes only (never labels after ``t``), and missing windows
    remain missing.
    """
    x = pd.DataFrame(index=panel.index)

    # Gold history: limited forward fill so short gaps become usable, while
    # long gaps remain missing rather than imputed.
    price = panel.y.ffill(limit=cfg["gold_feature_ffill_limit"])
    x["gold_latest"] = price
    x["gold_missing"] = panel.y.isna().astype(int)

    log_price = np.log(price.where(price > 0))
    for lag in [1, 2, 5, 10, 20, 60]:
        x[f"gold_lag_{lag}"] = price.shift(lag)
    for window in [5, 20, 60]:
        x[f"gold_ret_{window}"] = log_price.diff(window)
        x[f"gold_vol_{window}"] = (
            log_price.diff().rolling(window, min_periods=window).std()
        )
        x[f"gold_mean_ratio_{window}"] = (
            price / price.rolling(window, min_periods=window).mean() - 1
        )

    # External levels, missingness, and staleness age.
    roles = [
        "brent",
        "brent_spot",
        "wti",
        "yield10",
        "dxy",
        "fed",
        "cpi_yoy",
        "cpi_mom",
        "gpr",
        "gpr_acts",
        "gpr_threats",
        "gpr_ma7",
        "gpr_acts_ma7",
        "gpr_threats_ma7",
    ]
    for name in roles:
        x[name] = panel[name]
        x[name + "_missing"] = panel[name].isna().astype(int)
        x[name + "_age_days"] = panel[name + "_age_days"]

    # Trailing changes: log differences for positive price series, absolute
    # differences for rates/indices (and negative WTI prices).
    for name in ["brent", "brent_spot", "dxy"]:
        for lag in [5, 20]:
            x[f"{name}_ret_{lag}"] = np.log(panel[name].where(panel[name] > 0)).diff(
                lag
            )
    for name in ["wti", "yield10", "fed"]:
        for lag in [5, 20]:
            x[f"{name}_chg_{lag}"] = panel[name].diff(lag)

    # Policy (yield minus Fed) and calendar features known at the origin.
    x["yield_spread"] = panel.yield10 - panel.fed
    x["weekday"] = x.index.dayofweek
    x["year_sin"] = np.sin(2 * np.pi * x.index.dayofyear / 365.25)
    x["year_cos"] = np.cos(2 * np.pi * x.index.dayofyear / 365.25)
    return x.replace([np.inf, -np.inf], np.nan)


def feature_columns(x: pd.DataFrame, group: str, model: str) -> list[str]:
    """Pick the candidate feature set for a model family and feature group.

    Group meanings
    --------------
    - ``gold_only``: a model must rely purely on gold history/calendar inputs.
    - ``external``: adds the preselected external variables.

    Models that require sequences or fixed column order trim to compact sets;
    Prophet never receives gold-derived regressors because future gold values
    are unknown at an origin.
    """
    if model == "prophet":
        gold = []
        extra = [
            "dxy",
            "fed",
            "yield10",
            "cpi_yoy",
            "gpr_ma7",
            "brent_spot",
            "wti",
        ]
    elif model in ("gru", "lstm"):
        gold = [
            "gold_latest",
            "gold_missing",
            "gold_ret_5",
            "gold_ret_20",
            "gold_vol_20",
            "year_sin",
            "year_cos",
            "weekday",
        ]
        extra = [
            "dxy",
            "fed",
            "yield10",
            "cpi_yoy",
            "cpi_mom",
            "gpr_ma7",
            "gpr_acts_ma7",
            "gpr_threats_ma7",
            "brent_spot",
            "wti",
            "yield_spread",
            "dxy_age_days",
            "gpr_ma7_age_days",
            "wti_missing",
        ]
    else:
        gold = [
            c
            for c in x
            if c.startswith("gold_") or c in ("weekday", "year_sin", "year_cos")
        ]
        extra = [
            "dxy",
            "fed",
            "yield10",
            "cpi_yoy",
            "cpi_mom",
            "gpr_ma7",
            "gpr_acts_ma7",
            "gpr_threats_ma7",
            "brent_spot",
            "wti",
            "yield_spread",
            "dxy_age_days",
            "gpr_ma7_age_days",
            "wti_missing",
        ]
    return gold if group == "gold_only" else gold + extra


def feature_description(name: str) -> str:
    """Return a short human-readable description for one candidate feature."""
    if name.endswith("_age_days"):
        return (
            "Calendar days since the latest source observation; "
            "unavailable until assumed release."
        )
    if "missing" in name:
        return "Indicator: source value unavailable at this origin."
    if "_ret_" in name:
        return "Trailing log difference over the indicated weekday-grid steps."
    if "_chg_" in name:
        return "Trailing absolute difference over the indicated weekday-grid steps."
    if name.startswith("gold_"):
        return (
            "Gold history through origin t; filled inputs limited by "
            "configuration, labels never filled."
        )
    return "Origin-time value or deterministic calendar feature; see README and availability table."
