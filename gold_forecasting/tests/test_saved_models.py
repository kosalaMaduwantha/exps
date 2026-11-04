"""Real saved-bundle reload parity, with no fitting/CV/selection.

Run explicitly with GOLD_VERIFY_SAVED_MODELS=1 after installing modeling pins.
Normal API fixture tests do not require all heavy model backends.
"""

import os
import unittest

import numpy as np
import pandas as pd

from model_lib.artifacts.serialization import load_bundle
from model_lib.forecasting.predictor import predict_model
from model_lib.preprocessing import ROOT, load_prepared


@unittest.skipUnless(
    os.environ.get("GOLD_VERIFY_SAVED_MODELS") == "1",
    "Explicit saved-model integration check",
)
class SavedModelParityTests(unittest.TestCase):
    """Compare each native loaded bundle to its saved notebook forecast rows."""

    def test_all_four_frozen_bundles(self) -> None:
        panel, features, _ = load_prepared()
        for kind in ("prophet", "xgboost", "gru", "lstm"):
            with self.subTest(kind=kind):
                directory = ROOT / "artifacts" / kind
                expected = pd.read_csv(directory / "future_forecast.csv")
                bundle = load_bundle(directory / "final_model")
                origin = int(panel.index.get_loc(pd.Timestamp(expected.origin.iloc[0])))
                paths = (
                    pd.read_csv(directory / "future_regressors.csv")
                    if kind == "prophet"
                    else None
                )
                actual = predict_model(
                    bundle,
                    panel.iloc[: origin + 1],
                    features.iloc[: origin + 1],
                    np.array([origin]),
                    bundle["training_config"],
                    paths,
                )[0]
                np.testing.assert_allclose(
                    actual, expected.forecast, rtol=1e-5, atol=1e-3
                )
