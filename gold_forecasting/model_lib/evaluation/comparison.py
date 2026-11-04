"""Gold evaluation / comparison. Extracted default daily algorithms; see function contracts.

Model-fitting workflows are invoked only by notebooks, never by serving.
"""

from __future__ import annotations


import pandas as pd
from model_lib.settings import ROOT


def compare_results(
    stage: str = "cv",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare saved model forecasts on a common set of eligible targets.

    Only origins/targets present in every model's table are pooled together,
    so rankings do not depend on models having unequal missing-label coverage.
    Persistence and trailing-drift baselines are appended to the comparison.
    """
    from model_lib.evaluation.metrics import metrics

    files = sorted((ROOT / "artifacts").glob(f"*/{stage}_predictions.csv"))
    if not files:
        raise FileNotFoundError("No prediction files yet. Run model notebooks first.")

    frames = [
        pd.read_csv(path, parse_dates=["origin", "target_date"]) for path in files
    ]
    all_predictions = pd.concat(frames, ignore_index=True)

    keys = ["fold", "origin", "target_date", "horizon"]
    eligible = [
        frame.dropna(subset=["actual", "prediction"]).set_index(keys).index
        for frame in frames
    ]
    common = eligible[0]
    for index in eligible[1:]:
        common = common.intersection(index)
    comparison = (
        all_predictions.set_index(keys)
        .loc[lambda d: d.index.isin(common)]
        .reset_index()
    )
    if comparison.empty:
        raise ValueError("No common eligible predictions across models.")

    # Append the deterministic persistence and drift baselines for context.
    baseline = comparison.drop_duplicates(keys).copy()
    baseline["prediction"] = baseline.baseline
    baseline["model"] = "persistence"
    drift = comparison.drop_duplicates(keys).copy()
    drift["prediction"] = drift["drift"]
    drift["model"] = "trailing_drift"
    comparison = pd.concat([comparison, baseline, drift], ignore_index=True)

    summary = metrics(comparison)
    coverage = all_predictions.groupby("model").agg(
        total_predictions=("prediction", "size"),
        observed_targets=("actual", "count"),
    )
    coverage["common_predictions"] = len(common)

    destination = ROOT / "artifacts/comparison"
    destination.mkdir(exist_ok=True)
    summary.to_csv(destination / f"{stage}_metrics_common_targets.csv", index=False)
    coverage.to_csv(destination / f"{stage}_coverage.csv")
    return comparison, summary, coverage
