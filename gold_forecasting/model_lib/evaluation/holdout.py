"""Gold evaluation / holdout. Extracted default daily algorithms; see function contracts.

Model-fitting workflows are invoked only by notebooks, never by serving.
"""

from __future__ import annotations

import json

import pandas as pd
from model_lib.contracts import Config
from model_lib.settings import ROOT


def evaluate_holdout(
    kind: str,
    panel: pd.DataFrame,
    x: pd.DataFrame,
    folds: pd.DataFrame,
    cfg: Config,
    trials: list[Config] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Config]:
    """Evaluate the untouched final fold once, after locking parameters."""
    from model_lib.model_training.trainer import fit_model
    from model_lib.cross_validation.splitters import forecast_origins
    from model_lib.evaluation.metrics import metrics
    from model_lib.forecasting.predictor import predict_model
    from model_lib.evaluation.metrics import predictions_frame
    from model_lib.artifacts.serialization import save_bundle
    from model_lib.model_training.selection import select_parameters

    if not cfg["final_holdout_enabled"]:
        raise ValueError(
            "Review development results, then set final_holdout_enabled=true "
            "in config.json."
        )
    output = ROOT / "artifacts" / kind
    output.mkdir(parents=True, exist_ok=True)
    if (output / "holdout_predictions.csv").exists():
        raise FileExistsError(
            "Holdout already evaluated. Existing results are preserved; "
            "do not repeatedly tune on them."
        )

    fold = folds.loc[folds.fold.eq("holdout")].iloc[0]
    cutoff, stop = int(fold.cutoff), int(fold.stop)
    # Final parameter selection reproduces the latest development fold's inner
    # boundary; the untouched holdout is never used for choosing a configuration.
    development = folds.loc[folds.fold.ne("holdout")]
    if development.empty:
        raise ValueError("At least one development fold is required before holdout.")
    selection_cutoff = int(development.iloc[-1].cutoff)
    params, selection = select_parameters(kind, panel, x, selection_cutoff, cfg, trials)

    # Persist the selected parameters before any holdout score is revealed.
    (output / "locked_parameters.json").write_text(json.dumps(params, indent=2))
    selection.to_csv(output / "final_inner_selection.csv", index=False)

    bundle = fit_model(kind, panel, x, cutoff, params, cfg)
    origins = forecast_origins(panel, cutoff, stop, cfg)
    values = predict_model(bundle, panel, x, origins, cfg)
    predictions = predictions_frame(
        panel, origins, values, kind, "holdout", cutoff, cfg
    )
    predictions.to_csv(output / "holdout_predictions.csv", index=False)
    metrics(predictions).to_csv(output / "holdout_metrics.csv", index=False)
    save_bundle(bundle, output / "holdout_model")
    return predictions, metrics(predictions), params
