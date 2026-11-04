"""API schema v1. Dates are ISO calendar dates; prices use metadata target units.

All floating-point fields reject NaN and infinity. Optional intervals and metrics
are explicitly nullable. Unknown request fields are rejected, including regressors
not declared by the selected model. No modeling or server dependencies live here.
"""

from datetime import date, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

ModelId = Literal["prophet", "xgboost", "gru", "lstm"]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
Identifier = Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,99}$")]


class Contract(BaseModel):
    """Strict schema base: JSON-safe finite numbers and forbidden unknown fields."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: Literal[1] = 1


class Regressor(Contract):
    """One exact external feature name, not a gold lag or calendar input."""

    name: str
    description: str
    unit: str | None = None
    numeric_type: Literal["float"] = "float"
    required: Literal[True] = True


class ModelMetadata(Contract):
    """Published immutable bundle identity and inference input requirements."""

    model_id: ModelId
    version: Identifier
    recipe: str
    training_cutoff: date
    horizons: list[PositiveInt] = Field(min_length=1)
    feature_columns: list[str]
    external_regressors: list[Regressor]
    currency: Literal["USD"] = "USD"
    target_unit: str
    calendar: Literal["weekday_grid"] = "weekday_grid"
    lookback: PositiveInt = 1
    config_hash: str
    data_hash: str
    interval_support: bool = False
    regressor_mode: Literal["origin_held"] = "origin_held"

    @model_validator(mode="after")
    def valid_features(self) -> "ModelMetadata":
        """Unique horizons/features and exact external descriptor coverage are required."""
        if len(set(self.horizons)) != len(self.horizons) or len(
            set(self.feature_columns)
        ) != len(self.feature_columns):
            raise ValueError("Duplicate horizons or feature names.")
        expected = {
            name
            for name in self.feature_columns
            if not name.startswith("gold_")
            and name not in ("weekday", "year_sin", "year_cos")
        }
        names = [item.name for item in self.external_regressors]
        if set(names) != expected or len(names) != len(expected):
            raise ValueError(
                "Regressor descriptors must cover exactly the external feature schema."
            )
        return self


class ModelList(Contract):
    """Currently promoted models, never an implicit ranking."""

    items: list[ModelMetadata]


class SpotRequest(Contract):
    """Origin-time scenario; external inputs override only this request's row."""

    model_id: ModelId
    expected_version: Identifier
    origin_date: date
    horizon: PositiveInt
    regressors: dict[str, FiniteFloat] = Field(default_factory=dict)


class ForecastRequest(Contract):
    """Same origin-time scenario for all horizons in the bundle."""

    model_id: ModelId
    expected_version: Identifier
    origin_date: date
    regressors: dict[str, FiniteFloat] = Field(default_factory=dict)


class ForecastPoint(Contract):
    """One configured-horizon price forecast, never an interpolated daily row."""

    target_date: date
    horizon: PositiveInt
    prediction: FiniteFloat
    lower: FiniteFloat | None = None
    upper: FiniteFloat | None = None
    interval_level: FiniteFloat | None = Field(default=None, gt=0, lt=1)

    @model_validator(mode="after")
    def valid_interval(self) -> "ForecastPoint":
        """Intervals must have both bounds, an explicit level, and ordered limits."""
        values = (self.lower, self.upper, self.interval_level)
        if any(v is not None for v in values):
            if any(v is None for v in values):
                raise ValueError("Intervals require lower, upper, and interval_level.")
            if (
                self.lower is not None
                and self.upper is not None
                and self.lower > self.upper
            ):
                raise ValueError("Interval lower exceeds upper.")
        return self


class ForecastResponse(Contract):
    """Inference response including enough context to label retrospective scenarios."""

    model_id: ModelId
    version: Identifier
    origin_date: date
    origin_price: FiniteFloat
    training_cutoff: date
    target_unit: str
    currency: Literal["USD"] = "USD"
    regressor_mode: str = "origin_held"
    retrospective: bool
    regressors: dict[str, FiniteFloat]
    points: list[ForecastPoint]


class Page(Contract):
    """Stable offset page; total describes the filtered population before slicing."""

    total: int = Field(ge=0)
    offset: int = Field(ge=0)
    limit: int = Field(gt=0, le=2000)


class Origins(Page):
    """Eligible observed origins; latest is for the full filtered population."""

    version: Identifier
    items: list[date]
    latest: date | None


class HistoryRow(Contract):
    """Unobserved prices remain null even if features were forward-filled."""

    date: date
    price: FiniteFloat | None
    observed: bool


class History(Page):
    """Prepared target history, with an immutable source/config identity."""

    items: list[HistoryRow]
    data_hash: str
    target_unit: str


class Provenance(Contract):
    """Notebook-published report provenance; no inferred accuracy claims."""

    model_id: ModelId
    version: Identifier
    recipe: str
    config_hash: str
    data_hash: str
    training_cutoff: date
    assumptions: str
    created_at: datetime


class EvaluationRow(ForecastPoint):
    """Frozen test prediction, corresponding target, and saved forecast error."""

    origin_date: date
    actual: FiniteFloat
    baseline: FiniteFloat | None = None
    residual: FiniteFloat


class HorizonMetrics(Contract):
    """Saved metrics for the full report/horizon, not the current display page."""

    horizon: PositiveInt
    n: PositiveInt
    mae: FiniteFloat
    rmse: FiniteFloat
    baseline_mae: FiniteFloat | None = None
    skill_vs_persistence: FiniteFloat | None = None


class EvaluationReport(Contract):
    """Immutable test/holdout report; filters never recompute these metrics."""

    report_id: Identifier
    provenance: Provenance
    split: Literal["test", "holdout"]
    target_unit: str
    rows: list[EvaluationRow] = Field(min_length=1)
    metrics: list[HorizonMetrics] = Field(min_length=1)
    metric_scope: Literal["full_report_per_horizon"] = "full_report_per_horizon"


class EvaluationSummary(Contract):
    """Small report selector payload; model identity is in provenance."""

    report_id: Identifier
    provenance: Provenance
    split: Literal["test", "holdout"]
    start_date: date
    end_date: date
    horizons: list[int]
    sample_count: int


class EvaluationList(Contract):
    """Available immutable test reports, including historical model versions."""

    items: list[EvaluationSummary]


class EvaluationPage(Page):
    """Filtered prediction rows plus explicitly full-report saved metrics."""

    report_id: Identifier
    provenance: Provenance
    split: Literal["test", "holdout"]
    target_unit: str
    items: list[EvaluationRow]
    metrics: list[HorizonMetrics]
    metric_scope: Literal["full_report_per_horizon"] = "full_report_per_horizon"


class ForecastSnapshot(Contract):
    """Notebook-produced future forecasts; retrieval never executes inference."""

    snapshot_id: Identifier
    provenance: Provenance
    origin_date: date
    origin_price: FiniteFloat
    target_unit: str
    points: list[ForecastPoint] = Field(min_length=1)


class SnapshotSummary(Contract):
    """Small immutable forecast snapshot selector payload."""

    snapshot_id: Identifier
    provenance: Provenance
    origin_date: date
    horizons: list[int]


class SnapshotList(Contract):
    """Available published snapshots, newest origin first."""

    items: list[SnapshotSummary]


class ErrorResponse(Contract):
    """Uniform machine-readable errors; details do not expose local paths."""

    code: str
    message: str


class Health(Contract):
    """Liveness or readiness status with human-readable diagnostic."""

    status: Literal["ok", "unavailable"]
    message: str
