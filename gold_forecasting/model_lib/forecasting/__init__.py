"""Frozen-bundle inference. Importing this package never fits a model."""

from .predictor import predict_model

__all__ = ["predict_model"]
