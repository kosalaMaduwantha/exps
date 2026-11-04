"""Prepared-data and immutable-bundle inference service, independent of HTTP.

Requests operate on private feature copies. All shared fitted state is read-only;
an inference lock protects native backends and promotion resolves immutable keys.
"""

from datetime import date
from pathlib import Path
from threading import RLock
from typing import Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from model_lib.contracts import Bundle

from gold_contracts.schemas import (
    ForecastPoint,
    ForecastRequest,
    ForecastResponse,
    ModelMetadata,
)
from model_lib.artifacts.registry import (
    ArtifactStore,
    ArtifactUnavailable,
    config_digest,
    prepared_digest,
)
from model_lib.preprocessing import ROOT, load_prepared


class InvalidRequest(ValueError):
    """Request is invalid for the selected bundle or prepared context."""


class VersionConflict(ValueError):
    """A form uses metadata for a version that is no longer active."""


class ForecastService:
    """Lazy data/bundle cache with injectable loaders for no-training tests.

    Args:
        store: Immutable serving artifact store.
        project_root: Root containing prepared data and the source manifest.
        prepared_loader: Optional aligned panel/features/folds loader for tests.
        bundle_loader: Optional native deserializer for tests.
        predictor: Optional native prediction function for tests.
    """

    def __init__(
        self,
        store: ArtifactStore,
        project_root: Path = ROOT,
        *,
        prepared_loader: Callable[
            [], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ] = load_prepared,
        bundle_loader: Callable[[Path], Bundle] | None = None,
        predictor: Callable[..., NDArray[np.float64]] | None = None,
    ) -> None:
        self.store = store
        self.project_root = project_root
        self._prepared_loader = prepared_loader
        self._bundle_loader = bundle_loader
        self._predictor = predictor
        self._bundles: dict[tuple[str, str], Bundle] = {}
        self._lock = RLock()

    def prepared(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Validate source/config freshness each call; never silently reuse stale data."""
        try:
            return self._prepared_loader()
        except (OSError, ValueError, KeyError, AssertionError) as exc:
            raise ArtifactUnavailable(
                "Prepared data is unavailable or stale; rerun notebook 01."
            ) from exc

    def metadata(self, model_id: str) -> ModelMetadata:
        """Return active typed metadata from the registry."""
        return self.store.metadata(model_id)

    def _bundle(self, metadata: ModelMetadata) -> Bundle:
        directory = self.store.verify_bundle(metadata)
        key = (metadata.model_id, metadata.version)
        if key not in self._bundles:
            try:
                if self._bundle_loader is None:
                    from model_lib.artifacts.serialization import load_bundle

                    bundle = load_bundle(directory)
                else:
                    bundle = self._bundle_loader(directory)
                if (
                    bundle["columns"] != metadata.feature_columns
                    or bundle["training_config"]["horizons"] != metadata.horizons
                    or bundle["kind"] != metadata.model_id
                    or bundle["params"].get("lookback", 1) != metadata.lookback
                    or bundle["training_config"]["target_unit"] != metadata.target_unit
                    or config_digest(bundle["training_config"]) != metadata.config_hash
                    or bundle["training_cutoff"] != metadata.training_cutoff.isoformat()
                ):
                    raise ValueError("Incompatible native metadata.")
                self._bundles[key] = bundle
            except Exception as exc:
                raise ArtifactUnavailable(
                    "Model backend or native bundle is unavailable."
                ) from exc
        return self._bundles[key]

    def eligible(
        self, metadata: ModelMetadata, panel: pd.DataFrame, features: pd.DataFrame
    ) -> list[date]:
        """Observed origins with enough family-specific context; no target lookahead."""
        missing = set(metadata.feature_columns) - set(features.columns)
        if missing:
            raise ArtifactUnavailable(
                "Prepared features do not match the model schema."
            )
        valid = panel.y.notna().to_numpy().copy()
        if metadata.model_id in ("gru", "lstm"):
            valid[: metadata.lookback - 1] = False
            long_gap = panel.y.isna().rolling(6, min_periods=6).sum().ge(6)
            for index in range(metadata.lookback - 1, len(panel)):
                if long_gap.iloc[index - metadata.lookback + 1 : index + 1].any():
                    valid[index] = False
        elif metadata.model_id == "xgboost":
            valid[:60] = False
        return [stamp.date() for stamp in panel.index[valid]]

    def predict(
        self, request: ForecastRequest, horizon: int | None = None
    ) -> ForecastResponse:
        """Run frozen inference; overrides affect only this request's origin row.

        Recurrent overrides replace the final timestep, not historical context.
        Prophet holds supplied external values constant; no auxiliary models fit.
        Raises InvalidRequest, VersionConflict, or ArtifactUnavailable.
        """
        with self._lock:
            metadata = self.metadata(request.model_id)
            if request.expected_version != metadata.version:
                raise VersionConflict(
                    "Active model changed; refresh model metadata and submit again."
                )
            if horizon is not None and horizon not in metadata.horizons:
                raise InvalidRequest("Unsupported horizon.")
            expected = {item.name for item in metadata.external_regressors}
            if set(request.regressors) != expected:
                raise InvalidRequest(
                    f"Exact external regressor names required: {sorted(expected)}"
                )
            panel, features, _ = self.prepared()
            if prepared_digest(self.project_root) != metadata.data_hash:
                raise ArtifactUnavailable(
                    "Published bundle does not match prepared data identity."
                )
            if request.origin_date not in self.eligible(metadata, panel, features):
                raise InvalidRequest(
                    "Origin must be an observed weekday with eligible model context."
                )
            origin = int(panel.index.get_loc(pd.Timestamp(request.origin_date)))
            # Work only through the origin; no future rows enter inference transforms.
            context = features.iloc[: origin + 1].copy()
            for name, value in request.regressors.items():
                context.loc[context.index[-1], name] = value
            bundle = self._bundle(metadata)
            try:
                if self._predictor is None:
                    from model_lib.forecasting.predictor import predict_model

                    values = predict_model(
                        bundle,
                        panel.iloc[: origin + 1],
                        context,
                        np.array([origin]),
                        bundle["training_config"],
                    )[0]
                else:
                    values = self._predictor(
                        bundle,
                        panel.iloc[: origin + 1],
                        context,
                        np.array([origin]),
                        bundle["training_config"],
                    )[0]
                if (
                    len(values) != len(metadata.horizons)
                    or not np.isfinite(values).all()
                ):
                    raise ValueError("Invalid model output.")
            except Exception as exc:
                raise ArtifactUnavailable("Frozen model inference failed.") from exc
            points = [
                ForecastPoint(
                    target_date=(
                        pd.Timestamp(request.origin_date) + pd.offsets.BDay(step)
                    ).date(),
                    horizon=step,
                    prediction=float(value),
                )
                for step, value in zip(metadata.horizons, values)
                if horizon is None or step == horizon
            ]
            return ForecastResponse(
                model_id=metadata.model_id,
                version=metadata.version,
                origin_date=request.origin_date,
                origin_price=float(panel.y.iloc[origin]),
                training_cutoff=metadata.training_cutoff,
                target_unit=metadata.target_unit,
                retrospective=request.origin_date < metadata.training_cutoff,
                regressors=request.regressors,
                points=points,
            )

    def ready(self) -> None:
        """Validate prepared data and deserialize all active bundles without fitting."""
        with self._lock:
            self.prepared()
            active = self.store.active_versions()
            if not active:
                raise ArtifactUnavailable("No models have been promoted.")
            for model_id in active:
                metadata = self.metadata(model_id)
                if metadata.data_hash != prepared_digest(self.project_root):
                    raise ArtifactUnavailable("Active model data identity mismatch.")
                self._bundle(metadata)
