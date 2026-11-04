"""Non-training helpers. Training, CV, selection and refits run only in notebooks."""

import argparse

from model_lib.preprocessing import load_config, preprocess


def main() -> None:
    """Prepare data or compare saved results; never fit a model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["preprocess", "compare"], required=True)
    args = parser.parse_args()
    if args.stage == "preprocess":
        panel, _, _, _ = preprocess(load_config())
        print(
            f"Prepared {len(panel)} weekday slots; {panel.y.notna().sum()} observed targets."
        )
    else:
        from model_lib.evaluation.comparison import compare_results

        _, table, coverage = compare_results()
        print(coverage.to_string())
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
