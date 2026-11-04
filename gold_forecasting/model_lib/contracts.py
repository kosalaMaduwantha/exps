"""Research library types. DataFrames retain documented model-specific schemas.

Backend objects are the explicit Any boundary: Prophet, sklearn, Keras and
XGBoost expose different native fit/predict/save interfaces. They are not HTTP
payloads and must only be deserialized from trusted local publications.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeAlias, TypedDict, NotRequired

import numpy as np
import pandas as pd
from numpy.typing import NDArray

Config: TypeAlias = dict[str, Any]


class Bundle(TypedDict):
    """Native fitted state; only backend-specific objects use Any.

    fit_model attaches training_config/cutoff before serialization. A Prophet
    bundle has model/medians, trees have models, and recurrent bundles also own
    preprocessor/target_scaler. Fields remain distinct rather than forcing a
    shared backend object representation.
    """

    kind: str
    columns: list[str]
    params: Config
    training_config: NotRequired[Config]
    training_cutoff: NotRequired[str]
    model: NotRequired[Any]
    models: NotRequired[list[Any]]
    medians: NotRequired[pd.Series]
    preprocessor: NotRequired[Any]
    target_scaler: NotRequired[Any]
    best_epochs: NotRequired[int]
    history: NotRequired[dict[str, list[float]]]


@dataclass(frozen=True)
class PreparedData:
    """Aligned weekday-indexed data; panel.y is observed-only, features are causal."""

    panel: pd.DataFrame
    features: pd.DataFrame
    folds: pd.DataFrame
    config: Config


class ModelAdapter(Protocol):
    """Small explicit adapter contract, preserving native backend state in bundles."""

    def fit(self, data: PreparedData, cutoff: int, parameters: Config) -> Bundle:
        """Fit using only eligible observations through the inclusive cutoff."""
        ...

    def predict(
        self, bundle: Bundle, data: PreparedData, origins: NDArray[np.int64]
    ) -> NDArray[np.float64]:
        """Return positive prices shaped (origin_count, horizon_count), without fitting."""
        ...

    def save(self, bundle: Bundle, directory: Path) -> None:
        """Save model-specific preprocessing and native state to a trusted directory."""
        ...

    def load(self, directory: Path) -> Bundle:
        """Restore state or raise for corrupt/incompatible metadata."""
        ...
