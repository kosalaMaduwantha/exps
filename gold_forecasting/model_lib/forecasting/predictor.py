"""Model-specific frozen inference and inverse price transformations.

Feature order and scalers are owned by the saved bundle. Returned arrays have
shape (number of origins, configured horizon count), in original price units.
No auxiliary regressor estimation or fitted transformations occur here.
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from model_lib.contracts import Bundle, Config


def windows(array: NDArray, origins: NDArray, lookback: int) -> NDArray:
    """Stack origin-ending (batch, lookback, feature) windows as float32 inputs."""
    return np.stack([array[o - lookback + 1 : o + 1] for o in origins]).astype(
        "float32"
    )


def predict_model(
    bundle: Bundle,
    panel: pd.DataFrame,
    x: pd.DataFrame,
    origins: NDArray,
    cfg: Config,
    future_regressors: pd.DataFrame | None = None,
) -> NDArray:
    """Predict fixed-origin prices; optional precomputed paths require one origin.

    Args:
        bundle: Trusted loaded native model and fitted preprocessing state.
        panel: Weekday-indexed target panel with observed prices in y.
        x: Same-index causal features, in the bundle's saved column schema.
        origins: Integer positions into panel/x, with eligible trailing context.
        cfg: Training-compatible configuration, including identical horizons.
        future_regressors: Optional Prophet paths indexed by date or containing ds.
    Returns:
        Price array shaped (len(origins), len(cfg['horizons'])).
    Raises:
        ValueError: Horizons differ or paths accompany multiple origins.
    """
    columns = bundle["columns"]
    if bundle["training_config"]["horizons"] != cfg["horizons"]:
        raise ValueError("Prediction horizons differ from the trained model.")
    kind = bundle["kind"]
    anchor = panel.y.iloc[origins].to_numpy()
    if future_regressors is not None and len(origins) != 1:
        raise ValueError("Future regressor paths require exactly one origin.")
    if kind == "xgboost":
        returns = np.column_stack(
            [model.predict(x.iloc[origins][columns]) for model in bundle["models"]]
        )
        return anchor[:, None] * np.exp(returns)
    if kind in ("gru", "lstm"):
        transformed = bundle["preprocessor"].transform(x[columns])
        inputs = windows(transformed, origins, bundle["params"]["lookback"])
        standardized = bundle["model"](inputs, training=False).numpy()
        returns = bundle["target_scaler"].inverse_transform(standardized)
        return anchor[:, None] * np.exp(returns)
    paths = None
    if future_regressors is not None:
        paths = future_regressors.copy()
        if "ds" in paths.columns:
            paths = paths.set_index("ds")
        paths.index = pd.to_datetime(paths.index)
    predictions = []
    for origin in origins:
        dates = pd.bdate_range(
            panel.index[origin] + pd.offsets.BDay(1), periods=max(cfg["horizons"])
        )
        future = pd.DataFrame({"ds": dates})
        for column in columns:
            value = x.iloc[origin][column]
            fallback = value if pd.notna(value) else bundle["medians"][column]
            if paths is not None and column in paths.columns:
                series = paths[column].reindex(dates)
                future[column] = series.where(series.notna(), fallback).to_numpy()
            else:
                future[column] = fallback
        forecast = bundle["model"].predict(future)
        predictions.append([np.exp(forecast.yhat.iloc[h - 1]) for h in cfg["horizons"]])
    return np.asarray(predictions)
