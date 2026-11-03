"""Focused regression checks for time alignment and missing-label safety.

Run with:
    python -m unittest -v test_pipeline.py

The tests are intentionally small: they check causality, staleness, fold
isolation, and label handling rather than model quality.
"""
import unittest

import numpy as np
import pandas as pd

from gold_common_utils import (
    available_series,
    forecast_origins,
    load_config,
    load_prepared,
    make_features,
    parse_gold_date,
)
from gold_model_utils import forecast_external_paths, sequence_rows, supervised_rows


class TimeSafetyTests(unittest.TestCase):
    """Guards around release timing and leakage between folds."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config()
        cls.panel, cls.x, cls.folds = load_prepared()

    def test_release_is_not_visible_early(self):
        """A value released on a weekend must already be visible afterward."""
        dates = pd.bdate_range('2024-01-25', '2024-02-06')
        values, age = available_series(
            [12], ['2024-01-01'], ['2024-02-03'], dates, 60
        )
        self.assertTrue(values.loc[:'2024-02-02'].isna().all())
        self.assertEqual(values.loc['2024-02-05'], 12)
        self.assertEqual(age.loc['2024-02-05'], 35)

    def test_stale_release_expires(self):
        """Values older than the staleness limit should become missing."""
        dates = pd.bdate_range('2024-01-01', periods=12)
        values, _age = available_series(
            [12], ['2024-01-01'], ['2024-01-01'], dates, 3
        )
        self.assertTrue(values.loc['2024-01-05':].isna().all())

    def test_feature_history_unchanged_by_future(self):
        """Appending future rows must not alter any earlier feature values."""
        cutoff = int(self.folds.iloc[0].cutoff)
        short = make_features(self.panel.iloc[:cutoff+1], self.cfg)
        long = make_features(self.panel, self.cfg).iloc[:cutoff+1]
        pd.testing.assert_frame_equal(short, long)

    def test_targets_never_cross_cutoff(self):
        """Tree training labels must be observed and stay before the cutoff."""
        cutoff = int(self.folds.iloc[0].cutoff)
        for horizon in self.cfg['horizons']:
            origins, target = supervised_rows(self.panel, cutoff, horizon)
            self.assertTrue((origins + horizon <= cutoff).all())
            self.assertTrue(self.panel.y.iloc[origins+horizon].notna().all())
            np.testing.assert_allclose(
                target,
                np.log(
                    self.panel.y.iloc[origins+horizon].to_numpy()
                    / self.panel.y.iloc[origins].to_numpy()
                ),
            )

    def test_sequence_labels_are_observed(self):
        """Recurrent windows must close before the cutoff and skip gaps."""
        cutoff = int(self.folds.iloc[0].cutoff)
        origins = sequence_rows(
            self.panel, cutoff, 60, self.cfg['horizons']
        )
        for horizon in self.cfg['horizons']:
            self.assertTrue(self.panel.y.iloc[origins+horizon].notna().all())
            self.assertTrue((origins + horizon <= cutoff).all())

    def test_holdout_not_in_development(self):
        """Every development fold must lie strictly inside the holdout cutoff."""
        holdout_cutoff = int(
            self.folds.loc[self.folds.fold.eq('holdout'), 'cutoff'].iloc[0]
        )
        for fold in self.folds.loc[self.folds.fold.ne('holdout')].itertuples():
            self.assertLessEqual(fold.stop, holdout_cutoff)
            origins = forecast_origins(
                self.panel, fold.cutoff, fold.stop, self.cfg
            )
            self.assertTrue((origins > fold.cutoff).all())
            self.assertTrue(
                (origins + max(self.cfg['horizons']) <= fold.stop).all()
            )

    def test_target_gaps_remain_missing(self):
        """Missing labels must stay missing; nothing is silently filled."""
        self.assertGreater(self.panel.y.isna().sum(), 0)
        self.assertTrue(
            self.panel.loc[self.panel.target_observed.eq(0), 'y'].isna().all()
        )

    def test_negative_wti_is_preserved_without_log_features(self):
        """Negative WTI must survive and only get one-sided changes."""
        self.assertLess(self.panel.wti.min(), 0)
        self.assertIn('wti_chg_5', self.x)
        self.assertNotIn('wti_ret_5', self.x)
        self.assertFalse(np.isinf(self.x.to_numpy()).any())

    def test_date_parser_does_not_guess_malformed_year(self):
        """Only documented date formats are repaired."""
        self.assertEqual(parse_gold_date('31.10..2017'), pd.Timestamp('2017-10-31'))
        self.assertTrue(pd.isna(parse_gold_date('22.08.20232')))
        self.assertTrue(pd.isna(parse_gold_date('01/00/1900')))


class FutureMacroPathTests(unittest.TestCase):
    """Guards around auxiliary future external-regressor paths."""

    def test_future_paths_are_anchored_and_ignore_future_rows(self):
        """Paths start at the origin value and never read rows after it."""
        index = pd.bdate_range('2024-01-01', periods=80)
        panel = pd.DataFrame(index=index)
        x = pd.DataFrame(index=index)
        x['macro'] = np.linspace(1.0, 2.0, len(index))
        x['flat'] = 5.0
        cfg = {'horizons': [1, 2, 3], 'horizon_labels': ['a', 'b', 'c']}
        origin = 60

        before = forecast_external_paths(panel, x, origin, ['macro', 'flat'], cfg)
        changed = x.copy()
        changed.iloc[origin + 1 :] = 999.0
        after = forecast_external_paths(
            panel, changed, origin, ['macro', 'flat'], cfg
        )

        expected_dates = pd.bdate_range(
            panel.index[origin] + pd.offsets.BDay(1), periods=3
        )
        self.assertEqual(len(before), 3)
        self.assertTrue(before.ds.isin(expected_dates).all())
        self.assertTrue(before['macro'].notna().all())
        self.assertTrue(before['flat'].eq(5.0).all())
        self.assertLessEqual(
            abs(before['macro'].iloc[0] - x['macro'].iloc[origin]), 0.1
        )
        pd.testing.assert_frame_equal(before, after)


if __name__ == '__main__':
    unittest.main()
