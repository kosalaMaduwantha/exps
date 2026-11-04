"""Typed HTTPX client. Parses every response and owns its connection lifecycle.

The optional transport supports in-process API integration tests. This module
depends only on HTTPX and lightweight contracts, never on server/model packages.
"""

from datetime import date
from threading import RLock
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError
from cachetools import TTLCache

from gold_contracts.schemas import (
    ErrorResponse,
    EvaluationList,
    EvaluationPage,
    ForecastResponse,
    ForecastSnapshot,
    History,
    ModelId,
    ModelList,
    ModelMetadata,
    Origins,
    SnapshotList,
    SpotRequest,
)

T = TypeVar("T", bound=BaseModel)
# Byte payloads, not client/session objects, are shared. Version and request
# filters are part of the key; explicit refresh invalidates all cached GETs.
ReadCacheKey = tuple[tuple[str, str], str, tuple[tuple[str, str | int], ...]]
_cache: TTLCache[ReadCacheKey, bytes] = TTLCache(maxsize=128, ttl=60)
_cache_lock = RLock()


def clear_read_cache() -> None:
    """Clear bounded read-only response caching after refresh or version conflicts."""
    with _cache_lock:
        _cache.clear()


class ApiError(RuntimeError):
    """Actionable client error, with optional HTTP status for conflict handling."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class ApiClient:
    """Context-managed validated API client with bounded network waits.

    Args:
        base_url: API origin, e.g. http://127.0.0.1:8000.
        transport: Optional HTTPX test transport.
    """

    def __init__(
        self,
        base_url: str,
        *,
        transport: httpx.BaseTransport | None = None,
        cache_version: str = "",
    ) -> None:
        self._cache_key = (base_url.rstrip("/"), cache_version)
        self._cache_enabled = transport is None
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            transport=transport,
            timeout=httpx.Timeout(30.0, connect=5.0),
            trust_env=False,
        )

    def __enter__(self) -> "ApiClient":
        return self

    def __exit__(self, *args: object) -> None:
        self._http.close()

    def _request(
        self,
        method: str,
        path: str,
        schema: type[T],
        *,
        params: dict[str, str | int] | None = None,
        body: BaseModel | None = None,
    ) -> T:
        cache_key = (self._cache_key, path, tuple(sorted((params or {}).items())))
        if method == "GET" and self._cache_enabled:
            with _cache_lock:
                cached = _cache.get(cache_key)
            if cached is not None:
                return schema.model_validate_json(cached)
        try:
            response = self._http.request(
                method,
                path,
                params=params,
                json=body.model_dump(mode="json") if body else None,
            )
        except httpx.HTTPError as exc:
            raise ApiError(
                "Cannot reach the API. Check its URL and that Uvicorn is running."
            ) from exc
        if response.is_error:
            try:
                error = ErrorResponse.model_validate_json(response.content)
                message = error.message
            except ValidationError:
                message = f"API returned HTTP {response.status_code}."
            raise ApiError(message, response.status_code)
        try:
            parsed = schema.model_validate_json(response.content)
            if method == "GET" and self._cache_enabled:
                with _cache_lock:
                    _cache[cache_key] = response.content
            return parsed
        except ValidationError as exc:
            raise ApiError(
                "API response does not match schema v1; check client/server versions."
            ) from exc

    def models(self) -> ModelList:
        """Return active model metadata used for all dashboard selectors."""
        return self._request("GET", "/models", ModelList)

    def model(self, model_id: ModelId) -> ModelMetadata:
        """Refresh one model's active metadata, notably after a 409 conflict."""
        return self._request("GET", f"/models/{model_id}", ModelMetadata)

    def origins(
        self, model_id: ModelId, *, offset: int = 0, limit: int = 500
    ) -> Origins:
        """Return descending eligible origins with the full latest eligible date."""
        return self._request(
            "GET",
            f"/models/{model_id}/origins",
            Origins,
            params={"offset": offset, "limit": limit},
        )

    def history(self, start_date: date, end_date: date, *, offset: int = 0) -> History:
        """Return an observed-history page; callers must follow total for full charts."""
        return self._request(
            "GET",
            "/history",
            History,
            params={
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "offset": offset,
                "limit": 2000,
            },
        )

    def evaluations(self, model_id: ModelId) -> EvaluationList:
        """Return published evaluation selectors, possibly empty."""
        return self._request("GET", f"/models/{model_id}/evaluations", EvaluationList)

    def evaluation(
        self, model_id: ModelId, report_id: str, horizon: int, *, offset: int = 0
    ) -> EvaluationPage:
        """Return one horizon's frozen rows and full-report-per-horizon metrics."""
        return self._request(
            "GET",
            f"/models/{model_id}/evaluations/{report_id}",
            EvaluationPage,
            params={"horizon": horizon, "offset": offset, "limit": 2000},
        )

    def snapshots(self, model_id: ModelId) -> SnapshotList:
        """Return saved future forecasts in newest-origin order."""
        return self._request(
            "GET", f"/models/{model_id}/forecast-snapshots", SnapshotList
        )

    def snapshot(self, model_id: ModelId, snapshot_id: str) -> ForecastSnapshot:
        """Read saved forecast points, never request training or inference."""
        return self._request(
            "GET",
            f"/models/{model_id}/forecast-snapshots/{snapshot_id}",
            ForecastSnapshot,
        )

    def spot(self, request: SpotRequest) -> ForecastResponse:
        """Submit one explicit origin-time scenario and return the resolved model version."""
        return self._request("POST", "/forecasts/spot", ForecastResponse, body=request)
