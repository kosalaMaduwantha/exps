"""Pure experimental recipe checks: explicit dependencies and chronological labels."""

import unittest

import numpy as np
import pandas as pd

from model_lib.recipes.legacy_monthly_prophet import month_start, monthly_mean
from model_lib.recipes.daily_prophet_experiment import monthly_asof, ccf_table
from model_lib.recipes.daily_xgboost_challenger import build_direct_dataset


class RecipeTests(unittest.TestCase):
    """Exercise migrated helpers independently of their original notebook globals."""

    def test_monthly_aggregation(self) -> None:
        frame = pd.DataFrame(
            {
                "date": ["2024-01-02", "2024-01-20", "2024-02-03"],
                "value": [2.0, 4.0, 10.0],
            }
        )
        result = monthly_mean(frame, "date", "value")
        self.assertEqual(result.tolist(), [3.0, 10.0])
        self.assertEqual(
            month_start(["2024-03-17"]).iloc[0], pd.Timestamp("2024-03-01")
        )

    def test_release_alignment_and_lag_screening(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=80)
        monthly = pd.Series([5.0], index=pd.to_datetime(["2024-01-01"]))
        result = monthly_asof(monthly, 15, dates)
        self.assertTrue(result.loc[:"2024-02-14"].isna().all())
        self.assertEqual(result.loc["2024-02-15"], 5)
        frame = pd.DataFrame(
            {"target": np.arange(80), "feature": np.arange(80)}, index=dates
        )
        self.assertAlmostEqual(
            ccf_table(frame, "target", ["feature"], 0).iloc[0].r, 1.0
        )

    def test_challenger_direct_labels_are_purged(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=30)
        target = pd.Series(np.linspace(2000, 2100, 30), index=dates)
        frame = pd.DataFrame({"static": 1.0}, index=dates)
        rows = build_direct_dataset(
            ["static"],
            5,
            dates[20],
            1,
            DRIFT=pd.Series(0.0, index=dates),
            DYN_HIST=pd.DataFrame(index=dates),
            FUTURE_CORE=[],
            TARGET_CAL=[],
            F_hist=frame,
            bdays=dates,
            obs=target.notna(),
            target=target,
        )
        self.assertTrue((rows.target_ds < dates[20]).all())
        positions = dates.get_indexer(rows.target_ds)
        origins = dates.get_indexer(rows.origin_ds)
        np.testing.assert_allclose(
            rows.y,
            np.log(target.iloc[positions].to_numpy() / target.iloc[origins].to_numpy())
            / rows.h,
        )
