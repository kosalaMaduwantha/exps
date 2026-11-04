"""Gold forecasting / notebook. Extracted default daily algorithms; see function contracts.

Model-fitting workflows are invoked only by notebooks, never by serving.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from model_lib.contracts import Config
from model_lib.settings import ROOT


def refit_and_forecast(
    kind: str, panel: pd.DataFrame, x: pd.DataFrame, cfg: Config
) -> pd.DataFrame:
    """Refit the locked configuration and forecast future horizons once.

    The forecast starts from the latest observed gold quote. Prophet uses
    auxiliary daily as-of paths for its external regressors; other models keep
    origin-time features. Generates no calibrated uncertainty interval.
    """
    from model_lib.model_training.trainer import fit_model
    from model_lib.forecasting.regressors import forecast_external_paths
    from model_lib.forecasting.predictor import predict_model
    from model_lib.artifacts.serialization import save_bundle

    output = ROOT / "artifacts" / kind
    params_path = output / "locked_parameters.json"
    if not params_path.exists():
        raise FileNotFoundError("Run the final parameter lock and holdout stage first.")
    params = json.loads(params_path.read_text())
    cutoff = int(np.flatnonzero(panel.y.notna().to_numpy())[-1])
    bundle = fit_model(kind, panel, x, cutoff, params, cfg)

    future_regressors = None
    if kind == "prophet":
        future_regressors = forecast_external_paths(
            panel, x, cutoff, bundle["columns"], cfg
        )
        future_regressors.to_csv(output / "future_regressors.csv", index=False)

    values = predict_model(
        bundle,
        panel,
        x,
        np.array([cutoff]),
        cfg,
        future_regressors=future_regressors,
    )[0]
    save_bundle(bundle, output / "final_model")

    origin_price = float(panel.y.iloc[cutoff])
    forecast = pd.DataFrame(
        {
            "origin": panel.index[cutoff],
            "horizon": cfg["horizons"],
            "horizon_label": cfg["horizon_labels"],
            "target_date": [
                panel.index[cutoff] + pd.offsets.BDay(h) for h in cfg["horizons"]
            ],
            "origin_price": origin_price,
            "forecast": values,
            "change_from_origin_pct": (values / origin_price - 1) * 100,
            "model": kind,
        }
    )
    forecast.to_csv(output / "future_forecast.csv", index=False)
    return forecast
