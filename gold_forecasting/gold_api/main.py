"""FastAPI application factory and chart-data routes.

Run ``uvicorn gold_api.main:app`` from the project root. GOLD_ARTIFACT_ROOT
selects the local serving store. All publication remains an explicit notebook
operation. Import/startup creates no artifacts and performs no model fitting.
"""

import logging
import os
from datetime import date
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from gold_contracts.schemas import (
    ErrorResponse,
    EvaluationList,
    EvaluationPage,
    EvaluationSummary,
    ForecastRequest,
    ForecastResponse,
    ForecastSnapshot,
    Health,
    History,
    HistoryRow,
    ModelId,
    ModelList,
    ModelMetadata,
    Origins,
    SnapshotList,
    SnapshotSummary,
    SpotRequest,
)
from gold_api.service import ForecastService, InvalidRequest, VersionConflict
from model_lib.artifacts.registry import (
    ArtifactNotFound,
    ArtifactStore,
    ArtifactUnavailable,
    prepared_digest,
)
from model_lib.preprocessing import ROOT, load_config

Limit = Annotated[int, Query(ge=1, le=2000)]
Offset = Annotated[int, Query(ge=0)]
Identity = Annotated[str, Query(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,99}$")]
# FastAPI's OpenAPI response metadata is a third-party heterogeneous mapping.
ERRORS: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorResponse} for code in (404, 409, 422, 503)
}
logger = logging.getLogger(__name__)


def validate_dates(start: date | None, end: date | None) -> None:
    """Reject reversed inclusive date filters instead of returning misleading empties."""
    if start is not None and end is not None and start > end:
        raise InvalidRequest("start_date must not exceed end_date.")


def create_app(service: ForecastService | None = None) -> FastAPI:
    """Create an injectable application; tests supply small fixture-backed services."""
    if service is None:
        store = ArtifactStore(
            Path(os.environ.get("GOLD_ARTIFACT_ROOT", str(ROOT / "artifacts/serving")))
        )
        service = ForecastService(store)
    serving = service
    api = FastAPI(title="Gold Forecast API", version="1.0.0", responses=ERRORS)

    @api.exception_handler(ArtifactUnavailable)
    async def unavailable(_request: Request, exc: ArtifactUnavailable) -> JSONResponse:
        """Map missing/corrupt data to 503 without exposing backend file paths."""
        logger.warning("Artifact unavailable: %s", exc)
        return JSONResponse(
            status_code=503,
            content=ErrorResponse(code="unavailable", message=str(exc)).model_dump(),
        )

    @api.exception_handler(ArtifactNotFound)
    async def not_found(_request: Request, exc: ArtifactNotFound) -> JSONResponse:
        """Unknown model/report/snapshot identity is 404."""
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(code="not_found", message=str(exc)).model_dump(),
        )

    @api.exception_handler(VersionConflict)
    async def conflict(_request: Request, exc: VersionConflict) -> JSONResponse:
        """Metadata changed between form loading and inference submission."""
        return JSONResponse(
            status_code=409,
            content=ErrorResponse(
                code="version_conflict", message=str(exc)
            ).model_dump(),
        )

    @api.exception_handler(InvalidRequest)
    async def invalid(_request: Request, exc: InvalidRequest) -> JSONResponse:
        """Domain input validation follows the same typed error shape as schemas."""
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                code="invalid_request", message=str(exc)
            ).model_dump(),
        )

    @api.exception_handler(RequestValidationError)
    async def schema_invalid(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Avoid serializing non-finite invalid input inside validation error bodies."""
        fields = ", ".join(".".join(map(str, error["loc"])) for error in exc.errors())
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                code="invalid_request",
                message=f"Invalid fields: {fields}",
            ).model_dump(),
        )

    @api.get("/health", response_model=Health)
    def health() -> Health:
        """Liveness only; no disk reads or native model imports."""
        return Health(status="ok", message="API is running.")

    @api.get("/ready", response_model=Health)
    def ready() -> Health:
        """Readiness requires fresh prepared data and loadable promoted bundles."""
        serving.ready()
        return Health(
            status="ok", message="Prepared data and promoted models are ready."
        )

    @api.get("/models", response_model=ModelList)
    def models() -> ModelList:
        """List active model versions and exact external feature requirements."""
        return ModelList(
            items=[
                serving.metadata(model_id)
                for model_id in serving.store.active_versions()
            ]
        )

    @api.get("/models/{model_id}", response_model=ModelMetadata)
    def model(model_id: ModelId) -> ModelMetadata:
        """Return typed metadata used to construct inference forms."""
        return serving.metadata(model_id)

    @api.get("/models/{model_id}/origins", response_model=Origins)
    def origins(
        model_id: ModelId,
        start_date: date | None = None,
        end_date: date | None = None,
        horizon: int | None = None,
        limit: Limit = 500,
        offset: Offset = 0,
    ) -> Origins:
        """Page eligible observed origins in descending date order."""
        validate_dates(start_date, end_date)
        metadata = serving.metadata(model_id)
        if horizon is not None and horizon not in metadata.horizons:
            raise InvalidRequest("Unsupported horizon.")
        panel, features, _ = serving.prepared()
        rows = [
            day
            for day in reversed(serving.eligible(metadata, panel, features))
            if (start_date is None or day >= start_date)
            and (end_date is None or day <= end_date)
        ]
        return Origins(
            version=metadata.version,
            items=rows[offset : offset + limit],
            total=len(rows),
            latest=rows[0] if rows else None,
            limit=limit,
            offset=offset,
        )

    @api.get("/history", response_model=History)
    def history(
        start_date: date | None = None,
        end_date: date | None = None,
        limit: Limit = 500,
        offset: Offset = 0,
    ) -> History:
        """Page ascending prepared history with observed-only prices and data identity."""
        validate_dates(start_date, end_date)
        panel, _, _ = serving.prepared()
        rows = [
            HistoryRow(
                date=stamp.date(),
                price=None if pd.isna(row.y) else float(row.y),
                observed=bool(row.target_observed),
            )
            for stamp, row in panel.iterrows()
            if (start_date is None or stamp.date() >= start_date)
            and (end_date is None or stamp.date() <= end_date)
        ]
        return History(
            items=rows[offset : offset + limit],
            total=len(rows),
            offset=offset,
            limit=limit,
            data_hash=prepared_digest(serving.project_root),
            target_unit=load_config()["target_unit"],
        )

    @api.get("/models/{model_id}/evaluations", response_model=EvaluationList)
    def evaluations(model_id: ModelId) -> EvaluationList:
        """List frozen test reports; no results is a valid empty list."""
        return EvaluationList(
            items=[
                EvaluationSummary(
                    report_id=value.report_id,
                    provenance=value.provenance,
                    split=value.split,
                    start_date=min(row.target_date for row in value.rows),
                    end_date=max(row.target_date for row in value.rows),
                    horizons=sorted({row.horizon for row in value.rows}),
                    sample_count=len(value.rows),
                )
                for value in serving.store.evaluations(model_id)
            ]
        )

    @api.get(
        "/models/{model_id}/evaluations/{report_id}", response_model=EvaluationPage
    )
    def evaluation(
        model_id: ModelId,
        report_id: str,
        horizon: Annotated[int | None, Query(gt=0)] = None,
        start_date: date | None = None,
        end_date: date | None = None,
        limit: Limit = 500,
        offset: Offset = 0,
    ) -> EvaluationPage:
        """Filter frozen prediction rows; metrics remain full-report per-horizon values."""
        validate_dates(start_date, end_date)
        try:
            report = serving.store.evaluation(model_id, report_id)
        except ValueError as exc:
            raise InvalidRequest("Invalid report identifier.") from exc
        if horizon is not None and horizon not in {row.horizon for row in report.rows}:
            raise InvalidRequest("Horizon is not available in this report.")
        rows = sorted(
            [
                row
                for row in report.rows
                if (horizon is None or row.horizon == horizon)
                and (start_date is None or row.target_date >= start_date)
                and (end_date is None or row.target_date <= end_date)
            ],
            key=lambda row: (row.target_date, row.origin_date, row.horizon),
        )
        return EvaluationPage(
            report_id=report.report_id,
            provenance=report.provenance,
            split=report.split,
            target_unit=report.target_unit,
            items=rows[offset : offset + limit],
            metrics=[
                metric
                for metric in report.metrics
                if horizon is None or metric.horizon == horizon
            ],
            total=len(rows),
            offset=offset,
            limit=limit,
        )

    @api.get("/models/{model_id}/forecast-snapshots", response_model=SnapshotList)
    def snapshots(model_id: ModelId) -> SnapshotList:
        """List notebook-published snapshots, preserving historical originating versions."""
        return SnapshotList(
            items=[
                SnapshotSummary(
                    snapshot_id=value.snapshot_id,
                    provenance=value.provenance,
                    origin_date=value.origin_date,
                    horizons=[point.horizon for point in value.points],
                )
                for value in serving.store.snapshots(model_id)
            ]
        )

    @api.get(
        "/models/{model_id}/forecast-snapshots/{snapshot_id}",
        response_model=ForecastSnapshot,
    )
    def snapshot(model_id: ModelId, snapshot_id: str) -> ForecastSnapshot:
        """Read one saved snapshot without invoking inference or auxiliary fitting."""
        try:
            return serving.store.snapshot(model_id, snapshot_id)
        except ValueError as exc:
            raise InvalidRequest("Invalid snapshot identifier.") from exc

    @api.post("/forecasts/spot", response_model=ForecastResponse)
    def spot(request: SpotRequest) -> ForecastResponse:
        """Return one configured-horizon prediction using exact origin-time regressors."""
        common = ForecastRequest(**request.model_dump(exclude={"horizon"}))
        return serving.predict(common, request.horizon)

    @api.post("/forecasts", response_model=ForecastResponse)
    def forecasts(request: ForecastRequest) -> ForecastResponse:
        """Return all bundle horizons from one frozen origin-time scenario."""
        return serving.predict(request)

    return api


app = create_app()
