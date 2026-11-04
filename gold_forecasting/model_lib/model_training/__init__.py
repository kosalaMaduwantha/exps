"""Public model_training functions. Algorithms live in focused modules.

Notebook workflows call these interfaces directly; no legacy facade modules.
"""

from model_lib.model_training.xgboost import supervised_rows as supervised_rows
from model_lib.model_training.xgboost import fit_xgboost as fit_xgboost
from model_lib.model_training.prophet import fit_prophet as fit_prophet
from model_lib.model_training.recurrent import sequence_rows as sequence_rows
from model_lib.model_training.recurrent import build_network as build_network
from model_lib.model_training.recurrent import fit_recurrent as fit_recurrent
from model_lib.model_training.trainer import fit_model as fit_model
from model_lib.forecasting.regressors import (
    forecast_external_paths as forecast_external_paths,
)
from model_lib.model_training.selection import score_predictions as score_predictions
from model_lib.model_training.selection import select_parameters as select_parameters
from model_lib.cross_validation.walk_forward import (
    run_cross_validation as run_cross_validation,
)
from model_lib.evaluation.holdout import evaluate_holdout as evaluate_holdout
from model_lib.forecasting.notebook import refit_and_forecast as refit_and_forecast
from model_lib.evaluation.comparison import compare_results as compare_results
from model_lib.model_training.selection import DEFAULT_TRIALS as DEFAULT_TRIALS
from model_lib.forecasting.predictor import predict_model as predict_model
from model_lib.forecasting.predictor import windows as windows
from model_lib.artifacts.serialization import save_bundle as save_bundle
from model_lib.artifacts.serialization import load_bundle as load_bundle
