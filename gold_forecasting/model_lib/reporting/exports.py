"""Notebook-only publication helpers for existing saved bundles and result tables.

These functions publish results; they never train or evaluate models. Call after
reviewing a notebook's outputs. Required provenance is explicit, not inferred from
unrelated final-refit files. Original artifacts are left unchanged.
"""

from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from gold_contracts.schemas import (
    EvaluationReport,
    EvaluationRow,
    ForecastPoint,
    ForecastSnapshot,
    HorizonMetrics,
    ModelId,
    ModelMetadata,
    Provenance,
    Regressor,
)
from model_lib.artifacts.registry import ArtifactStore, config_digest, prepared_digest
from model_lib.preprocessing import feature_description


def publish_saved_model(
    project_root: Path,
    store: ArtifactStore,
    model_id: ModelId,
    version: str,
    *,
    recipe: str = "validated_daily",
    source: Path | None = None,
) -> ModelMetadata:
    """Publish an existing trusted final model, checking its data configuration.

    Args:
        project_root: Gold project directory containing data/processed.
        store: Destination serving artifact store.
        model_id: Supported family.
        version: New immutable identifier selected by the notebook author.
        recipe: Explicit modeling recipe; never a claim of validated accuracy.
        source: Saved native directory, default artifacts/{model_id}/final_model.
    Returns:
        Serving metadata. Publication is not promoted until store.promote is called.
    Raises:
        ValueError: Saved training/preparation configurations disagree.
        FileExistsError: Version already exists.
    """
    source = source or project_root / "artifacts" / model_id / "final_model"
    native = json.loads((source / "metadata.json").read_text())
    config = native["training_config"]
    snapshot = json.loads(
        (project_root / "data/processed/config_snapshot.json").read_text()
    )
    ignored = {"final_holdout_enabled", "max_epochs", "patience"}
    if any(
        config.get(key) != value
        for key, value in snapshot.items()
        if key not in ignored
    ):
        raise ValueError("Saved bundle does not match prepared data configuration.")
    external = [
        name
        for name in native["columns"]
        if not name.startswith("gold_")
        and name not in ("weekday", "year_sin", "year_cos")
    ]
    metadata = ModelMetadata(
        model_id=model_id,
        version=version,
        recipe=recipe,
        training_cutoff=native["training_cutoff"],
        horizons=config["horizons"],
        feature_columns=native["columns"],
        target_unit=config["target_unit"],
        external_regressors=[
            Regressor(name=name, description=feature_description(name))
            for name in external
        ],
        lookback=native["params"].get("lookback", 1),
        config_hash=config_digest(config),
        data_hash=prepared_digest(project_root),
    )
    store.publish_bundle(source, metadata)
    return metadata


def provenance_for(metadata: ModelMetadata, assumptions: str) -> Provenance:
    """Create timestamped provenance from the model that actually produced outputs."""
    return Provenance(
        model_id=metadata.model_id,
        version=metadata.version,
        recipe=metadata.recipe,
        training_cutoff=metadata.training_cutoff,
        config_hash=metadata.config_hash,
        data_hash=metadata.data_hash,
        assumptions=assumptions,
        created_at=datetime.now(timezone.utc),
    )


def publish_future_table(
    store: ArtifactStore,
    metadata: ModelMetadata,
    snapshot_id: str,
    table: pd.DataFrame,
    *,
    assumptions: str,
) -> ForecastSnapshot:
    """Publish current five-column forecast tables without rerunning inference.

    Table columns: origin, origin_price, target_date, horizon, forecast. One origin
    is required. Prices are in metadata.target_unit. No intervals are fabricated.
    """
    if table.empty or table.origin.nunique() != 1 or table.origin_price.nunique() != 1:
        raise ValueError("A snapshot requires exactly one origin and origin price.")
    if sorted(table.horizon.tolist()) != sorted(metadata.horizons):
        raise ValueError("Snapshot horizons do not match the bundle.")
    origin = pd.Timestamp(table.origin.iloc[0]).date()
    if any(
        pd.Timestamp(row.target_date).date()
        != (pd.Timestamp(origin) + pd.offsets.BDay(row.horizon)).date()
        for row in table.itertuples()
    ):
        raise ValueError("Snapshot target dates do not match weekday-grid horizons.")
    snapshot = ForecastSnapshot(
        snapshot_id=snapshot_id,
        provenance=provenance_for(metadata, assumptions),
        origin_date=origin,
        origin_price=float(table.origin_price.iloc[0]),
        target_unit=metadata.target_unit,
        points=[
            ForecastPoint(
                target_date=pd.Timestamp(row.target_date).date(),
                horizon=int(row.horizon),
                prediction=float(row.forecast),
            )
            for row in table.itertuples()
        ],
    )
    store.publish_snapshot(snapshot)
    return snapshot


def publish_test_table(
    store: ArtifactStore,
    metadata: ModelMetadata,
    report_id: str,
    table: pd.DataFrame,
    *,
    assumptions: str,
) -> EvaluationReport:
    """Freeze a single-model holdout table and compute saved metrics in the notebook.

    Columns: origin, target_date, horizon, actual, prediction, baseline. Missing
    actuals are excluded, never imputed. metadata must describe the holdout model,
    not a later full-history refit. Residual is prediction minus actual.
    """
    table = table.dropna(subset=["actual", "prediction"]).copy()
    if table.empty:
        raise ValueError("No observed test targets to publish.")
    rows = [
        EvaluationRow(
            origin_date=pd.Timestamp(row.origin).date(),
            target_date=pd.Timestamp(row.target_date).date(),
            horizon=int(row.horizon),
            actual=float(row.actual),
            prediction=float(row.prediction),
            baseline=None if pd.isna(row.baseline) else float(row.baseline),
            residual=float(row.prediction - row.actual),
        )
        for row in table.itertuples()
    ]
    metrics = []
    for horizon, group in table.groupby("horizon"):
        errors = group.prediction - group.actual
        baseline = (
            (group.baseline - group.actual).abs().mean()
            if group.baseline.notna().all()
            else None
        )
        metrics.append(
            HorizonMetrics(
                horizon=int(horizon),
                n=len(group),
                mae=float(errors.abs().mean()),
                rmse=float((errors.pow(2).mean()) ** 0.5),
                baseline_mae=baseline,
                skill_vs_persistence=float(1 - errors.abs().mean() / baseline)
                if baseline
                else None,
            )
        )
    report = EvaluationReport(
        report_id=report_id,
        provenance=provenance_for(metadata, assumptions),
        split="holdout",
        target_unit=metadata.target_unit,
        rows=rows,
        metrics=metrics,
    )
    store.publish_evaluation(report)
    return report
