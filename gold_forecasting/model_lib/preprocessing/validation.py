"""Gold preprocessing / validation. Extracted default daily algorithms; see function contracts.

Model-fitting workflows are invoked only by notebooks, never by serving.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from model_lib.contracts import Config


def parse_gold_date(value: Any) -> Any:
    """Parse known gold-file formats explicitly; never guess malformed dates.

    Supported inputs are pandas/numpy datetimes, Excel serial numbers,
    ``DD.MM.YYYY``, ``DD/MM/YYYY``, ISO 8601, and ``YYYYMMDD``. Values that do
    not match any known format return ``NaT`` so the cleaning audit can
    quarantine them for manual review.
    """
    if pd.isna(value):
        return pd.NaT
    # Datetime-like objects (including Excel readers) are used directly.
    if isinstance(value, (pd.Timestamp, np.datetime64)) or hasattr(value, "year"):
        return pd.Timestamp(value).normalize()
    # Integers/floats are Excel serial dates, not all-purpose day offsets.
    if isinstance(value, (int, float)):
        if 20000 <= value <= 60000:
            return pd.Timestamp("1899-12-30") + pd.Timedelta(days=float(value))
        return pd.NaT
    # Strings: normalize whitespace, non-breaking spaces, and repeated dots.
    value = str(value).strip().replace("\xa0", "")
    value = re.sub(r"\.{2,}", ".", value)
    for fmt in ("%d.%m.%Y", "%d/%m/%Y", "%Y-%m-%d", "%Y%m%d"):
        parsed = pd.to_datetime(value, format=fmt, errors="coerce")
        if pd.notna(parsed):
            return parsed.normalize()
    return pd.NaT


def clean_gold(raw_dir: Path, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean the gold workbook and return the target plus a row-level audit.

    Date reconciliation rules
    -------------------------
    - If both date columns are valid and agree, the later column wins.
    - If the columns disagree, the row is treated as unresolved.
    - If only one column supplies an in-range date, that value is used and the
      resolution is recorded in ``date_resolution``.

    Target rules
    ------------
    - Weekend rows and out-of-bounds prices are quarantined, not deleted.
    - Conflicting duplicate prices for one date keep none of the copies.
    - Exact duplicates keep the first source row.
    """
    from model_lib.preprocessing.loaders import numeric

    raw = pd.read_excel(raw_dir / "gold_data_new.xlsx", header=None).iloc[3:].copy()
    # One-based Excel row number helps humans trace every retained/dropped row.
    raw["source_row"] = raw.index + 1
    raw["date_a"] = raw[0].map(parse_gold_date)
    raw["date_b"] = raw[1].map(parse_gold_date)

    start = pd.Timestamp(cfg["date_start"])
    end = pd.Timestamp(cfg["date_end"])
    good_a = raw.date_a.between(start, end)
    good_b = raw.date_b.between(start, end)

    # Prefer the second date column; recover from the first only when needed.
    raw["ds"] = raw.date_b.where(good_b, raw.date_a.where(good_a))
    mismatch = good_a & good_b & raw.date_a.ne(raw.date_b)
    raw.loc[mismatch, "ds"] = pd.NaT
    raw["date_resolution"] = "second_date_column"
    raw.loc[~good_b & good_a, "date_resolution"] = "recovered_from_first_date_column"
    raw.loc[raw.ds.isna(), "date_resolution"] = "unresolved"

    # Target values: reject obvious invalid prices and weekend rows.
    raw["y"] = numeric(raw[cfg["target_column"]])
    low, high = cfg["target_bounds"]
    raw["reason"] = ""
    raw.loc[raw.ds.isna(), "reason"] = "unresolved_date"
    raw.loc[raw.ds.notna() & ~raw.y.between(low, high), "reason"] = (
        "missing_or_out_of_bounds_target"
    )
    raw.loc[raw.ds.notna() & (raw.ds.dt.dayofweek >= 5), "reason"] = (
        "weekend_requires_review"
    )

    # Quarantine conflicting duplicate prices; keep the first exact duplicate.
    eligible = raw.reason.eq("")
    distinct = raw.loc[eligible].groupby("ds").y.nunique()
    conflicts = distinct.index[distinct > 1]
    raw.loc[eligible & raw.ds.isin(conflicts), "reason"] = (
        "conflicting_duplicate_requires_review"
    )
    eligible = raw.reason.eq("")
    duplicate = raw.loc[eligible].duplicated("ds", keep="first")
    raw.loc[duplicate.index[duplicate], "reason"] = "exact_target_duplicate"

    clean = raw.loc[raw.reason.eq(""), ["ds", "y"]].sort_values("ds").set_index("ds")

    # Robust trailing diagnostics flag possible errors, but do not delete real
    # shocks. The window uses only past rows (``shift(1)``) before the flag.
    changes = clean.y.diff()
    median = changes.rolling(60, min_periods=20).median().shift(1)
    mad = (
        changes.rolling(60, min_periods=20)
        .apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)
        .shift(1)
    )
    clean["large_change_flag"] = (changes - median).abs() > 8 * 1.4826 * mad.replace(
        0, np.nan
    )

    audit = raw[
        ["source_row", 0, 1, "date_a", "date_b", "ds", "y", "reason", "date_resolution"]
    ].copy()
    audit.columns = [
        "source_row",
        "raw_date_a",
        "raw_date_b",
        "date_a",
        "date_b",
        "ds",
        "y",
        "reason",
        "date_resolution",
    ]
    return clean, audit


def validate_panel(panel: pd.DataFrame, x: pd.DataFrame) -> None:
    """Assert the panel/candidate contract before any model sees the data."""
    assert panel.index.is_unique and panel.index.is_monotonic_increasing
    assert (panel.index.dayofweek < 5).all()
    assert x.index.equals(panel.index)
    assert not np.isinf(x.to_numpy()).any()
    assert panel.loc[panel.target_observed.eq(1), "y"].notna().all()
