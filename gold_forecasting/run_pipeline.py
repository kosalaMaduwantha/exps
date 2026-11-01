"""Command-line alternative to the model notebooks.

Run ``python run_pipeline.py --help`` for usage. The notebooks remain the
recommended entry point because they expose the review/decision workflow with
explicit human checkpoints; this script mirrors the same stages for automation.
"""
import argparse
import os

# Limit CPU threads before importing optional numerical/model libraries.
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')

from gold_common_utils import load_config, load_prepared, preprocess
from gold_model_utils import (
    compare_results,
    evaluate_holdout,
    refit_and_forecast,
    run_cross_validation,
)


def main():
    """Dispatch one pipeline stage from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--stage',
        choices=['preprocess', 'cv', 'holdout', 'forecast', 'compare'],
        required=True,
    )
    parser.add_argument(
        '--model',
        choices=['prophet', 'xgboost', 'gru', 'lstm', 'all'],
        default='all',
    )
    args = parser.parse_args()
    cfg = load_config()

    if args.stage == 'preprocess':
        panel, _features, _audit, _inventory = preprocess(cfg)
        print(
            f'Prepared {len(panel)} weekday slots and '
            f'{panel.y.notna().sum()} observed targets.'
        )
        return

    if args.stage == 'compare':
        _predictions, table, coverage = compare_results()
        print(coverage.to_string())
        print(table.to_string(index=False))
        return

    panel, features, folds = load_prepared()
    kinds = ['prophet', 'xgboost', 'gru', 'lstm'] if args.model == 'all' else [args.model]

    for kind in kinds:
        if args.stage == 'cv':
            _predictions, table, _selections = run_cross_validation(
                kind, panel, features, folds, cfg
            )
            print(table.to_string(index=False))
        elif args.stage == 'holdout':
            _predictions, table, _params = evaluate_holdout(
                kind, panel, features, folds, cfg
            )
            print(table.to_string(index=False))
        else:
            forecast = refit_and_forecast(kind, panel, features, cfg)
            print(forecast.to_string(index=False))


if __name__ == '__main__':
    main()
