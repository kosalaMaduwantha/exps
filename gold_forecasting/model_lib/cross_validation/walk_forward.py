"""Gold cross validation / walk forward. Extracted default daily algorithms; see function contracts.

Model-fitting workflows are invoked only by notebooks, never by serving.
"""

from __future__ import annotations

import json
import time

import pandas as pd
from model_lib.contracts import Config
from model_lib.settings import ROOT


def run_cross_validation(
    kind: str,
    panel: pd.DataFrame,
    x: pd.DataFrame,
    folds: pd.DataFrame,
    cfg: Config,
    trials: list[Config] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run nested walk-forward development evaluation for one model kind.

    For each non-holdout fold:
    1. tune parameters on an earlier inner block,
    2. refit on the full training cutoff with the selected parameters,
    3. forecast the outer block from fixed origins,
    4. checkpoint predictions/selection/timing to disk after every fold.
    """
    from model_lib.model_training.trainer import fit_model
    from model_lib.cross_validation.splitters import forecast_origins
    from model_lib.evaluation.metrics import metrics
    from model_lib.forecasting.predictor import predict_model
    from model_lib.evaluation.metrics import predictions_frame
    from model_lib.model_training.selection import select_parameters

    output = ROOT / "artifacts" / kind
    output.mkdir(parents=True, exist_ok=True)
    predictions, selections, timing = [], [], []

    for fold in folds.loc[folds.fold.ne("holdout")].itertuples(index=False):
        started = time.perf_counter()
        print(
            f"{kind}: {fold.fold}; train through {panel.index[fold.cutoff].date()}",
            flush=True,
        )
        params, selection = select_parameters(kind, panel, x, fold.cutoff, cfg, trials)
        selection["fold"] = fold.fold
        selections.append(selection)

        bundle = fit_model(kind, panel, x, fold.cutoff, params, cfg)
        origins = forecast_origins(panel, fold.cutoff, fold.stop, cfg)
        values = predict_model(bundle, panel, x, origins, cfg)
        predictions.append(
            predictions_frame(panel, origins, values, kind, fold.fold, fold.cutoff, cfg)
        )

        timing.append(
            {
                "fold": fold.fold,
                "seconds_including_selection": time.perf_counter() - started,
                "parameters": json.dumps(params),
                "network_parameters": (
                    bundle["model"].count_params() if kind in ("gru", "lstm") else None
                ),
            }
        )
        pd.DataFrame(timing).to_csv(output / "training_log.csv", index=False)
        # Checkpointing after each fold means an interrupted run keeps a
        # complete, human-reviewable subset of results.
        pd.concat(predictions).to_csv(output / "cv_predictions.csv", index=False)
        pd.concat(selections).to_csv(output / "inner_selection.csv", index=False)

    predictions = pd.concat(predictions, ignore_index=True)
    result = metrics(predictions)
    result.to_csv(output / "cv_metrics.csv", index=False)
    return predictions, result, pd.concat(selections, ignore_index=True)
