"""Gold preprocessing / loaders. Extracted default daily algorithms; see function contracts.

Model-fitting workflows are invoked only by notebooks, never by serving.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from model_lib.contracts import Config
from model_lib.settings import ROOT


def load_config() -> Config:
    """Read the shared experiment configuration from config.json."""
    cfg = json.loads((ROOT / "config.json").read_text())
    if len(cfg.get("horizon_labels", [])) != len(cfg["horizons"]):
        raise ValueError("horizon_labels and horizons must have the same length.")
    return cfg


def numeric(values: pd.Series) -> pd.Series:
    """Coerce comma/percent-formatted source text into numeric values."""
    cleaned = (
        values.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def build_panel(
    raw_dir: Path, cfg: Config
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build one weekday panel containing the target and external inputs.

    Every external series is pulled through an assumed availability date.
    The availability dates are documented assumptions because historical
    release vintages are unavailable. They preserve causality for the
    snapshot-based backtests.
    """
    from model_lib.preprocessing.availability import available_series
    from model_lib.preprocessing.validation import clean_gold

    gold, audit = clean_gold(raw_dir, cfg)

    # Weekday calendar; horizons count these slots, not exchange sessions.
    calendar = pd.bdate_range(cfg["date_start"], cfg["date_end"], name="ds")
    panel = gold.reindex(calendar)
    panel["target_observed"] = panel.y.notna().astype(int)
    inventory = []

    def add(
        name: str,
        values: Any,
        dates: Any,
        frequency: str = "daily",
        lag: int = 1,
        max_age: int = 10,
    ) -> None:
        """Attach one series plus its age column and record its assumptions."""
        dates = pd.DatetimeIndex(pd.to_datetime(dates))
        if frequency == "monthly":
            # Month-end observation becomes available after the lag days.
            availability = dates.to_period("M").to_timestamp("M") + pd.Timedelta(
                days=lag
            )
        elif frequency == "daily":
            availability = dates + pd.offsets.BDay(lag)
        else:
            availability = dates + pd.Timedelta(days=lag)
        panel[name], panel[name + "_age_days"] = available_series(
            values, dates, availability, calendar, max_age
        )
        inventory.append(
            {
                "series": name,
                "frequency": frequency,
                "lag": lag,
                "staleness_limit_calendar_days": max_age,
                "availability_status": "assumed; historical vintages unavailable",
            }
        )

    # Daily market closes: Brent, WTI, and the 10-year yield.
    for name, filename in [
        ("brent", "brent_historical.csv"),
        ("wti", "wti_historical.csv"),
    ]:
        frame = pd.read_csv(raw_dir / filename)
        dates = pd.to_datetime(frame.Date, format="%A, %B %d, %Y")
        add(
            name,
            numeric(frame.Close),
            dates,
            lag=cfg["daily_market_lag_business_days"],
        )

    # Backup Brent instrument: the EIA spot series, kept for comparison.
    frame = pd.read_csv(raw_dir / "Europe_Brent_Spot_Price_FOB.csv", skiprows=4)
    add(
        "brent_spot",
        numeric(frame.iloc[:, 1]),
        pd.to_datetime(frame.iloc[:, 0], format="%m/%d/%Y"),
        lag=cfg["daily_market_lag_business_days"],
    )

    # US 10-year yield.
    frame = pd.read_csv(
        raw_dir / "United States 10-Year Bond Yield Historical Data-use.csv"
    )
    add(
        "yield10",
        numeric(frame.Price),
        pd.to_datetime(frame.Date, format="%m/%d/%Y"),
        lag=cfg["daily_market_lag_business_days"],
    )

    # US Dollar Index. The final monthly candle could be incomplete (month not
    # finished at the configured END), so it is dropped. its own lag is one
    # month-end plus one calendar day.
    frame = pd.read_csv(raw_dir / "US Dollar Index Historical Data-use.csv")
    dates = pd.to_datetime(frame.Date, format="%m/%d/%Y")
    complete = dates.dt.to_period("M") < pd.Timestamp(cfg["date_end"]).to_period("M")
    add(
        "dxy",
        numeric(frame.loc[complete, "Price"]),
        dates[complete],
        "monthly",
        lag=1,
        max_age=65,
    )

    # Fed funds: the CSV source and its near-duplicate are cross-checked, and
    # only one copy is used as a feature.
    frame = pd.read_csv(raw_dir / "FEDFUNDS.csv")
    duplicate = pd.read_csv(raw_dir / "US-interest-rates-use.csv")
    assert np.array_equal(frame.iloc[:, 0], duplicate.iloc[:, 0])
    assert np.allclose(frame.iloc[:, 1], duplicate.iloc[:, 1])
    add(
        "fed",
        numeric(frame.FEDFUNDS),
        pd.to_datetime(frame.observation_date),
        "monthly",
        cfg["fed_release_lag_calendar_days"],
        75,
    )

    # CPI: reindex to a complete monthly calendar before computing changes so
    # 12 steps always span twelve reference months even with a missing row.
    frame = pd.read_excel(
        raw_dir / "SeriesReport-20261028041515_7e01eb-use.xlsx", header=11
    )
    frame = frame.loc[
        frame.Period.astype(str).str.fullmatch(r"M(0[1-9]|1[0-2])")
    ].copy()
    dates = pd.to_datetime(
        frame.Year.astype(int).astype(str) + "-" + frame.Period.str[1:] + "-01"
    )
    cpi = pd.Series(numeric(frame.Value).to_numpy(), index=dates).sort_index()
    cpi = cpi.reindex(pd.date_range(cpi.index.min(), cpi.index.max(), freq="MS"))
    for name, values in [
        ("cpi_yoy", cpi.pct_change(12, fill_method=None) * 100),
        ("cpi_mom", cpi.pct_change(fill_method=None) * 100),
    ]:
        add(
            name,
            values,
            values.index,
            "monthly",
            cfg["cpi_release_lag_calendar_days"],
            80,
        )

    # GPR: daily levels and 7-day trailing means; both are released with a lag.
    frame = pd.read_excel(raw_dir / "data_gpr_daily_recent.xls")
    dates = pd.to_datetime(frame.Date)
    for column, name in [
        ("GPRD", "gpr"),
        ("GPRD_ACT", "gpr_acts"),
        ("GPRD_THREAT", "gpr_threats"),
    ]:
        values = numeric(frame[column])
        add(
            name,
            values,
            dates,
            "calendar_daily",
            cfg["gpr_release_lag_calendar_days"],
            20,
        )
        trailing = (
            pd.Series(values.to_numpy(), index=dates)
            .sort_index()
            .rolling("7D", min_periods=3)
            .mean()
        )
        add(
            name + "_ma7",
            trailing,
            trailing.index,
            "calendar_daily",
            cfg["gpr_release_lag_calendar_days"],
            20,
        )
    return panel, audit, pd.DataFrame(inventory)


def preprocess(
    cfg: Config | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full shared preprocessing and cache everything to disk."""
    from model_lib.preprocessing.features import feature_description
    from model_lib.preprocessing.features import make_features
    from model_lib.cross_validation.splitters import make_folds
    from model_lib.preprocessing.validation import validate_panel

    cfg = cfg or load_config()
    output = ROOT / "data/processed"
    output.mkdir(parents=True, exist_ok=True)
    panel, audit, inventory = build_panel(ROOT / "data/raw", cfg)
    x = make_features(panel, cfg)
    validate_panel(panel, x)

    panel.to_csv(output / "panel.csv", index_label="ds")
    x.to_csv(output / "features.csv", index_label="ds")
    audit.to_csv(output / "gold_cleaning_audit.csv", index=False)
    inventory.to_csv(output / "availability_assumptions.csv", index=False)
    make_folds(panel.index, cfg).to_csv(output / "folds.csv", index=False)
    pd.DataFrame(
        {
            "feature": x.columns,
            "meaning": [feature_description(c) for c in x],
        }
    ).to_csv(output / "feature_dictionary.csv", index=False)
    (output / "config_snapshot.json").write_text(json.dumps(cfg, indent=2))

    manifest = [
        {
            "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted((ROOT / "data/raw").iterdir())
        if path.is_file()
    ]
    pd.DataFrame(manifest).to_csv(output / "source_manifest.csv", index=False)
    return panel, x, audit, inventory


def load_prepared() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the cached processed data from notebook 01 and verify freshness.

    The local config must match the config.json snapshot used during
    preprocessing. Raw-file hashes must still match the manifest, otherwise
    the pipeline must be rerun before any model training.
    """
    from model_lib.preprocessing.validation import validate_panel

    out = ROOT / "data/processed"
    panel = pd.read_csv(out / "panel.csv", index_col="ds", parse_dates=True)
    x = pd.read_csv(out / "features.csv", index_col="ds", parse_dates=True)
    snapshot = json.loads((out / "config_snapshot.json").read_text())
    manifest = pd.read_csv(out / "source_manifest.csv")

    # Every raw file must be the same one used when notebook 01 was run.
    for row in manifest.itertuples(index=False):
        source = ROOT / "data/raw" / row.file
        if (
            not source.exists()
            or hashlib.sha256(source.read_bytes()).hexdigest() != row.sha256
        ):
            raise ValueError(
                "Raw data changed: rerun notebook 01 and rebuild model artifacts."
            )

    cfg = load_config()
    ignored = {"final_holdout_enabled", "max_epochs", "patience"}
    differing = [key for key in cfg if key not in ignored and snapshot[key] != cfg[key]]
    if differing:
        raise ValueError("Configuration changed: rerun notebook 01 before training.")
    if not panel.index.equals(x.index):
        raise ValueError("Processed panel and feature dates differ: rerun notebook 01.")
    validate_panel(panel, x)
    return panel, x, pd.read_csv(out / "folds.csv")
