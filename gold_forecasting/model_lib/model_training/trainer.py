"""Gold model training / trainer. Extracted default daily algorithms; see function contracts.

Model-fitting workflows are invoked only by notebooks, never by serving.
"""

from __future__ import annotations

import pandas as pd
from model_lib.contracts import Config, Bundle
from model_lib.randomness import seed_everything


def fit_model(
    kind: str,
    panel: pd.DataFrame,
    x: pd.DataFrame,
    cutoff: int,
    params: Config,
    cfg: Config,
) -> Bundle:
    """Dispatch to one model adapter and attach fold-level metadata."""
    from model_lib.model_training.prophet import fit_prophet
    from model_lib.model_training.recurrent import fit_recurrent
    from model_lib.model_training.xgboost import fit_xgboost

    seed_everything(cfg["seed"])
    if kind == "xgboost":
        bundle = fit_xgboost(panel, x, cutoff, params, cfg)
    elif kind == "prophet":
        bundle = fit_prophet(panel, x, cutoff, params, cfg)
    elif kind in ("gru", "lstm"):
        bundle = fit_recurrent(kind, panel, x, cutoff, params, cfg)
    else:
        raise ValueError(f"Unknown model kind: {kind}")
    bundle["training_config"] = dict(cfg)
    bundle["training_cutoff"] = str(panel.index[cutoff].date())
    return bundle
