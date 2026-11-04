"""Gold preprocessing / availability. Extracted default daily algorithms; see function contracts.

Model-fitting workflows are invoked only by notebooks, never by serving.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def available_series(
    values: Any,
    dates: Any,
    available_dates: Any,
    calendar: pd.DatetimeIndex,
    max_age_days: int,
) -> tuple[pd.Series, pd.Series]:
    """Backward as-of join of one release series onto the weekday calendar.

    Returns the carried values and the age (calendar days) between each
    calendar row and the underlying observation date. Availability dates may
    fall on weekends, so the join uses the last release up to each origin.
    Values vanish once they exceed ``max_age_days`` so stale releases cannot
    silently substitute for fresh data.
    """
    releases = pd.DataFrame(
        {
            "observation_date": pd.to_datetime(dates),
            "available_date": pd.to_datetime(available_dates),
            "value": np.asarray(values, dtype=float),
        }
    ).dropna()
    releases = releases.sort_values("available_date").drop_duplicates(
        "available_date", keep="last"
    )
    joined = pd.merge_asof(
        pd.DataFrame({"ds": calendar}),
        releases,
        left_on="ds",
        right_on="available_date",
        direction="backward",
    )
    age = (joined.ds - joined.observation_date).dt.days
    value = joined.value.where(age <= max_age_days)
    return (
        pd.Series(value.to_numpy(), index=calendar),
        pd.Series(age.to_numpy(), index=calendar),
    )
