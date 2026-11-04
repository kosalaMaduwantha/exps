"""Trusted local model-specific serialization. No training or plotting imports."""

import json
from pathlib import Path

import joblib

from model_lib.contracts import Bundle


def save_bundle(bundle: Bundle, directory: Path | str) -> None:
    """Persist native state, column order, config, cutoff, and fitted transforms.

    Files are written into directory; immutable serving publication is a separate
    notebook action. Loading pickled preprocessing is restricted to trusted stores.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {
        "kind": bundle["kind"],
        "columns": bundle["columns"],
        "params": bundle["params"],
        "training_config": bundle["training_config"],
        "training_cutoff": bundle["training_cutoff"],
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2))
    if bundle["kind"] == "prophet":
        from prophet.serialize import model_to_json

        (directory / "prophet.json").write_text(model_to_json(bundle["model"]))
        joblib.dump(bundle["medians"], directory / "medians.joblib")
    elif bundle["kind"] == "xgboost":
        for horizon_id, model in enumerate(bundle["models"]):
            model.save_model(directory / f"horizon_{horizon_id}.json")
    else:
        bundle["model"].save(directory / "network.keras")
        joblib.dump(
            {
                "preprocessor": bundle["preprocessor"],
                "target_scaler": bundle["target_scaler"],
            },
            directory / "preprocessing.joblib",
        )
        (directory / "learning_history.json").write_text(json.dumps(bundle["history"]))


def load_bundle(directory: Path | str) -> Bundle:
    """Restore model state without fitting; raise for unknown kinds or missing files."""
    directory = Path(directory)
    bundle = json.loads((directory / "metadata.json").read_text())
    if bundle["kind"] == "prophet":
        from prophet.serialize import model_from_json

        bundle["model"] = model_from_json((directory / "prophet.json").read_text())
        bundle["medians"] = joblib.load(directory / "medians.joblib")
    elif bundle["kind"] == "xgboost":
        from xgboost import XGBRegressor

        files = sorted(
            directory.glob("horizon_*.json"),
            key=lambda path: int(path.stem.split("_")[-1]),
        )
        if len(files) != len(bundle["training_config"]["horizons"]):
            raise ValueError("Expected exactly one tree model per horizon.")
        bundle["models"] = []
        for path in files:
            model = XGBRegressor()
            model.load_model(path)
            bundle["models"].append(model)
    elif bundle["kind"] in ("gru", "lstm"):
        import tensorflow as tf

        bundle["model"] = tf.keras.models.load_model(directory / "network.keras")
        bundle.update(joblib.load(directory / "preprocessing.joblib"))
    else:
        raise ValueError("Unknown model family.")
    return bundle
