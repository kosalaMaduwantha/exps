"""Gold model training / selection. Extracted default daily algorithms; see function contracts.

Model-fitting workflows are invoked only by notebooks, never by serving.
"""

from __future__ import annotations

import json

import pandas as pd
from model_lib.contracts import Config

DEFAULT_TRIALS = {
    "prophet": [
        {"group": "gold_only", "changepoint_prior_scale": 0.05},
        {"group": "external", "changepoint_prior_scale": 0.05},
        {"group": "external", "changepoint_prior_scale": 0.2},
    ],
    "xgboost": [
        {"group": "gold_only", "max_depth": 2, "n_estimators": 250},
        {"group": "external", "max_depth": 2, "n_estimators": 250},
        {"group": "external", "max_depth": 3, "n_estimators": 400},
    ],
    "gru": [
        {"group": "gold_only", "lookback": 20, "units": 32},
        {"group": "external", "lookback": 20, "units": 32},
        {"group": "external", "lookback": 60, "units": 32},
    ],
    "lstm": [
        {"group": "gold_only", "lookback": 20, "units": 32},
        {"group": "external", "lookback": 20, "units": 32},
        {"group": "external", "lookback": 60, "units": 32},
    ],
}


def score_predictions(predictions: pd.DataFrame) -> float:
    """Report a single comparable score per fold on observed labels only."""
    from model_lib.evaluation.metrics import metrics

    table = metrics(predictions)
    if len(table) == 0:
        raise ValueError("No observed targets available for scoring.")
    return table.groupby("horizon").MAE.mean().mean()


def select_parameters(
    kind: str,
    panel: pd.DataFrame,
    x: pd.DataFrame,
    cutoff: int,
    cfg: Config,
    trials: list[Config] | None = None,
) -> tuple[Config, pd.DataFrame]:
    """Tune the model configuration on an earlier inner block only.

    Reviewers must see the inner scores before the outer block is ever used.
    This keeps the outer data reserved strictly for reporting.
    """
    from model_lib.model_training.trainer import fit_model
    from model_lib.cross_validation.splitters import forecast_origins
    from model_lib.forecasting.predictor import predict_model
    from model_lib.evaluation.metrics import predictions_frame

    inner_cutoff = cutoff - cfg["validation_steps"]
    origins = forecast_origins(panel, inner_cutoff, cutoff, cfg)
    if len(origins) == 0:
        raise ValueError("No eligible inner-validation origins.")

    records = []
    candidates = trials or DEFAULT_TRIALS[kind]
    for candidate_id, params in enumerate(candidates):
        bundle = fit_model(kind, panel, x, inner_cutoff, params, cfg)
        values = predict_model(bundle, panel, x, origins, cfg)
        predictions = predictions_frame(
            panel, origins, values, kind, "inner", inner_cutoff, cfg
        )
        score = score_predictions(predictions)
        records.append(
            {
                "candidate": candidate_id,
                "inner_MAE": score,
                "params": json.dumps(params),
            }
        )
        print(
            f"  candidate {candidate_id + 1}/{len(candidates)}: inner MAE={score:.3f}",
            flush=True,
        )

    table = pd.DataFrame(records)
    best = int(table.loc[table.inner_MAE.idxmin(), "candidate"])
    return dict(candidates[best]), table
