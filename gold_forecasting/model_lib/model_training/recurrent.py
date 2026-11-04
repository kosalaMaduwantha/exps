"""Gold model training / recurrent. Extracted default daily algorithms; see function contracts.

Model-fitting workflows are invoked only by notebooks, never by serving.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from model_lib.contracts import Config, Bundle

from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def sequence_rows(
    panel: pd.DataFrame, cutoff: int, lookback: int, horizons: list[int]
) -> np.ndarray:
    """Return eligible recurrent-training origins and their lookback windows.

    An origin is eligible only when the whole trailing window and every target
    horizon are observed. Long gaps are never bridged with fill or imputation,
    so a network never trains on an imputed gold history.
    """
    origins = np.arange(max(60, lookback - 1), cutoff - max(horizons) + 1)
    valid = panel.y.iloc[origins].notna().to_numpy(copy=True)
    for horizon in horizons:
        valid &= panel.y.iloc[origins + horizon].notna().to_numpy()

    # Reject any window that crosses at least a `lookback`-length target gap.
    long_gap = panel.y.isna().rolling(6, min_periods=6).sum().ge(6)
    for i, origin in enumerate(origins):
        valid[i] &= not long_gap.iloc[origin - lookback + 1 : origin + 1].any()
    return origins[valid]


def build_network(
    kind: str, lookback: int, n_features: int, n_outputs: int, units: int
) -> Any:
    """Assemble the small shared GRU/LSTM architecture."""
    import tensorflow as tf

    recurrent = tf.keras.layers.GRU if kind == "gru" else tf.keras.layers.LSTM
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(lookback, n_features)),
            recurrent(units, dropout=0.1),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(n_outputs),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
        loss="mse",
    )
    return model


def fit_recurrent(
    kind: str,
    panel: pd.DataFrame,
    x: pd.DataFrame,
    cutoff: int,
    params: Config,
    cfg: Config,
) -> Bundle:
    """Fit one recurrent model on this fold and refit fresh at the best epoch."""
    from model_lib.preprocessing.features import feature_columns
    from model_lib.forecasting.predictor import windows
    import tensorflow as tf

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(cfg["seed"])
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass

    columns = feature_columns(x, params["group"], kind)
    lookback = params["lookback"]
    origins = sequence_rows(panel, cutoff, lookback, cfg["horizons"])
    if len(origins) < 100:
        raise ValueError("Insufficient complete target windows for recurrent training.")

    # Epoch selection uses a chronological tail inside this training fold only;
    # training origins are purged so their targets cannot reach that tail.
    split_origin = origins[int(len(origins) * 0.85)]
    train_origins = origins[origins + max(cfg["horizons"]) < split_origin]
    val_origins = origins[origins >= split_origin]

    preprocessor = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
    )
    preprocessor.fit(x.iloc[:split_origin][columns])
    transformed = preprocessor.transform(x.iloc[: cutoff + 1][columns])
    anchor = panel.y.to_numpy()

    def labels(rows: np.ndarray) -> np.ndarray:
        """Log changes from each origin to every configured horizon."""
        return np.column_stack(
            [np.log(anchor[rows + h] / anchor[rows]) for h in cfg["horizons"]]
        )

    target_scaler = StandardScaler().fit(labels(train_origins))
    model = build_network(
        kind, lookback, transformed.shape[1], len(cfg["horizons"]), params["units"]
    )
    # Explicit batches avoid silent validation_slice boundary issues and keep
    # the chronological split observable.
    train_x = windows(transformed, train_origins, lookback)
    val_x = windows(transformed, val_origins, lookback)
    train_y = target_scaler.transform(labels(train_origins)).astype("float32")
    val_y = target_scaler.transform(labels(val_origins)).astype("float32")
    history = model.fit(
        train_x,
        train_y,
        validation_data=(val_x, val_y),
        epochs=cfg["max_epochs"],
        batch_size=32,
        shuffle=False,
        verbose=0,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=cfg["patience"],
                restore_best_weights=True,
            )
        ],
    )
    best_epochs = int(np.argmin(history.history["val_loss"]) + 1)

    # Reinitialize and refit on all eligible rows with the chosen epoch count.
    # No outer-validation labels or scaler statistics are ever reused here.
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(cfg["seed"])
    preprocessor.fit(x.iloc[: cutoff + 1][columns])
    transformed = preprocessor.transform(x.iloc[: cutoff + 1][columns])
    target_scaler.fit(labels(origins))
    model = build_network(
        kind, lookback, transformed.shape[1], len(cfg["horizons"]), params["units"]
    )
    model.fit(
        windows(transformed, origins, lookback),
        target_scaler.transform(labels(origins)).astype("float32"),
        epochs=best_epochs,
        batch_size=32,
        shuffle=False,
        verbose=0,
    )
    return {
        "kind": kind,
        "model": model,
        "columns": columns,
        "preprocessor": preprocessor,
        "target_scaler": target_scaler,
        "params": params,
        "best_epochs": best_epochs,
        "history": history.history,
    }
