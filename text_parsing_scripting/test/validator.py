"""Randomly compare original statement files with rows in the extractor CSV outputs.

Run from the project root, where the ``statement_parser`` package is importable:
    python test/validator.py ./statements ./statements_parsed_csv
    python test/validator.py ./statements ./statements_parsed_csv --samples 5 --seed 42

The second positional argument is the OUTPUT DIRECTORY produced by
data_extractor.py (transactions.csv / statement_info.csv / manifest.jsonl).

Large CSVs are read in chunks, so validation memory stays bounded no matter
how big the outputs are.

This is a consistency/regression check: it re-runs the current parser for each
sample and compares that result with the saved CSV rows. It does not prove the
parser's interpretation is independently correct.
"""

from __future__ import annotations

import argparse
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from statement_parser.csv_io import load_checkpoint
from statement_parser.parsing import discover_statement_files, parse_statement

# Amount-like strings such as "1,234.50", "707,841", "1976.0", or "123.45-"
# are normalized to numbers so cell types don't cause false failures. Plain
# (un-grouped) numbers are accepted because CSV values always arrive as text;
# the 1..12 digit cap keeps 16-digit card numbers out of the numeric branch.
_NUMERIC_TEXT = re.compile(
    r"^-?(?:\d{1,3}(?:,\d{3})*|\d{1,12})(?:\.\d{1,2})?-?$"
)
_CARD_NUMBER = re.compile(r"(?<!\d)(?:\d{4}[ -]?){3}\d{4}(?!\d)")

# transaction rows are read in chunks so huge outputs validate with bounded RAM
_CHUNK_ROWS = 1_000_000


def normalize(value: Any) -> Any:
    """Make equivalent pandas/CSV values comparable."""
    if value is None:
        return None
    try:
        missing = pd.isna(value)
        if isinstance(missing, bool) and missing:
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            return None
        return ("num", round(number, 2))

    text = re.sub(r"\s+", " ", str(value)).strip()
    if not text:
        return None
    if _NUMERIC_TEXT.fullmatch(text):
        negative = text.startswith("-") or text.endswith("-")
        cleaned = text.strip("-").replace(",", "")
        try:
            number = float(cleaned)
            return ("num", round(-number if negative else number, 2))
        except ValueError:
            pass
    return ("str", text)


def display_value(column: str, value: Any) -> str:
    """Format diagnostic values while masking full card numbers."""
    # Row-diff diagnostics pass normalized (kind, value) tuples; unwrap them
    # before rendering, while ordinary field diagnostics pass raw values.
    if isinstance(value, tuple) and len(value) == 2 and value[0] in {"num", "str", "bool"}:
        value = value[1]
    text = "<blank>" if normalize(value) is None else str(value)
    text = _CARD_NUMBER.sub(
        lambda m: "**** **** **** " + re.sub(r"\D", "", m.group(0))[-4:],
        text,
    )
    return text


def source_key(value: Any) -> str:
    """Convert a Source_File cell to the same POSIX path format as the parser."""
    return str(value).replace("\\", "/").strip() if normalize(value) is not None else ""


def records_counter(rows: list[dict[str, Any]], columns: list[str]) -> Counter:
    """Counter preserves duplicate rows instead of collapsing identical transactions."""
    return Counter(
        tuple((col, normalize(row.get(col))) for col in columns)
        for row in rows
    )


def get_rows_for_source(df: pd.DataFrame, source: str) -> list[dict[str, Any]]:
    if "Source_File" not in df.columns:
        raise ValueError("CSV output has no 'Source_File' column")
    mask = df["Source_File"].map(source_key) == source
    selected = df.loc[mask]
    return selected.to_dict(orient="records")


def load_rows_for_sources(csv_path: Path, sources: list[str]) -> pd.DataFrame:
    """Rows of a CSV whose Source_File is in `sources` (chunked reads)."""
    wanted = {s for s in sources if s}
    frames = []
    for chunk in pd.read_csv(csv_path, chunksize=_CHUNK_ROWS, dtype=object):
        frames.append(chunk[chunk["Source_File"].map(source_key).isin(wanted)])
    if not frames:
        return pd.DataFrame(columns=["Source_File"])
    return pd.concat(frames, ignore_index=True)


def compare_statement(path: Path, source: str, outputs: dict[str, pd.DataFrame],
                      manifest_latest: dict[str, dict[str, Any]]) -> bool:
    print(f"\n=== Checking: {source} ===")

    record = manifest_latest.get(source)
    if record is not None and record.get("status") != "ok":
        print(f"FAIL: source is flagged as error in the manifest: "
              f"{record.get('error')}")
        print("      fix the file and rerun the extractor; it will be retried automatically")
        return False

    try:
        header, expected_txns, _other_lines, _header_lines = parse_statement(path)
    except Exception as exc:
        print(f"FAIL: parser raised {type(exc).__name__}: {exc}")
        return False

    info_df = outputs.get("statement_info")
    txn_df = outputs.get("transactions")
    if info_df is None or txn_df is None:
        print("FAIL: output must contain 'statement_info.csv' and 'transactions.csv' rows")
        return False

    try:
        actual_info_rows = get_rows_for_source(info_df, source)
        actual_txns = get_rows_for_source(txn_df, source)
    except ValueError as exc:
        print(f"FAIL: {exc}")
        return False

    passed = True

    # Compare the header fields extracted directly by parse_statement().
    if len(actual_info_rows) != 1:
        print(
            f"FAIL: expected exactly 1 statement-info row for this file; "
            f"found {len(actual_info_rows)}"
        )
        passed = False
    else:
        actual_info = actual_info_rows[0]
        field_errors = []
        for field, expected_value in header.items():
            if field not in info_df.columns:
                field_errors.append((field, expected_value, "<column missing>"))
            elif normalize(expected_value) != normalize(actual_info.get(field)):
                field_errors.append((field, expected_value, actual_info.get(field)))
        if field_errors:
            passed = False
            print(f"FAIL: {len(field_errors)} statement-info field mismatch(es):")
            for field, expected, actual in field_errors[:20]:
                print(
                    f"  {field}: parser={display_value(field, expected)!r}; "
                    f"csv={display_value(field, actual)!r}"
                )
            if len(field_errors) > 20:
                print(f"  ... and {len(field_errors) - 20} more")
        else:
            print(f"PASS: all {len(header)} parser header fields match")

    # Compare transactions as a multiset so row order does not matter, but
    # repeated identical transactions must still appear the right number of times.
    expected_rows = [{"Source_File": source, **txn} for txn in expected_txns]
    all_columns = sorted(
        set(txn_df.columns).union(*(row.keys() for row in expected_rows))
        if expected_rows
        else set(txn_df.columns)
    )
    expected_counts = records_counter(expected_rows, all_columns)
    actual_counts = records_counter(actual_txns, all_columns)
    expected_only = expected_counts - actual_counts
    actual_only = actual_counts - expected_counts

    print(f"Transactions expected: {len(expected_rows)}; found in csv: {len(actual_txns)}")
    if expected_only or actual_only:
        passed = False
        print("FAIL: transaction rows do not match.")
        for label, differences in (("Missing from csv", expected_only), ("Unexpected in csv", actual_only)):
            shown = 0
            for row_tuple, count in differences.items():
                row = dict(row_tuple)
                fields = [
                    f"{col}={display_value(col, val)!r}"
                    for col, val in row.items()
                    if val is not None
                ]
                print(f"  {label} (count {count}): " + ", ".join(fields[:8]))
                shown += 1
                if shown >= 5:
                    break
            if sum(differences.values()) > shown:
                print(f"  ... plus {sum(differences.values()) - shown} additional row difference(s)")
    else:
        print("PASS: all transaction rows match (including duplicate counts)")

    print(f"Result: {'PASS' if passed else 'FAIL'}")
    return passed


def build_file_list(source_path: Path, pattern: str, out_dir: Path) -> list[tuple[Path, str]]:
    if source_path.is_file():
        return [(source_path, source_path.name)]
    if not source_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {source_path}")

    found = discover_statement_files(source_path, pattern=pattern)
    out_resolved = out_dir.resolve()
    if source_path.resolve() in out_resolved.parents:
        # never validate files inside the extractor's own output directory
        found = [p for p in found if out_resolved not in p.resolve().parents]
    return [
        (path, path.relative_to(source_path).as_posix())
        for path in found
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Randomly check original statement(s) against extractor CSV outputs."
    )
    parser.add_argument("source", help="original .txt file or directory containing statement files")
    parser.add_argument("outdir", help="output directory produced by data_extractor.py")
    parser.add_argument("--pattern", default="*.txt", help="statement glob for directory mode (default: %(default)s)")
    parser.add_argument("--samples", type=int, default=1, help="number of random files to check (default: 1)")
    parser.add_argument("--seed", type=int, default=None, help="optional random seed for repeatable selection")
    args = parser.parse_args(argv)

    if args.samples < 1:
        parser.error("--samples must be at least 1")

    source_path = Path(args.source).resolve()
    out_dir = Path(args.outdir).resolve()
    manifest_path = out_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        print(f"ERROR: no manifest at {manifest_path} (is {out_dir} an extractor output directory?)")
        return 2
    if not (out_dir / "transactions.csv").is_file():
        print(f"ERROR: no transactions.csv in {out_dir}")
        return 2

    try:
        files = build_file_list(source_path, args.pattern, out_dir)
    except Exception as exc:
        print(f"ERROR discovering statement files: {exc}")
        return 2

    if not files:
        print(f"ERROR: no statement files found under {source_path}")
        return 2

    manifest_latest = load_checkpoint(manifest_path)[0]

    rng = random.Random(args.seed)
    selected = rng.sample(files, k=min(args.samples, len(files)))
    print(f"Output directory: {out_dir}")
    print(f"Available source files: {len(files)}; selected for testing: {len(selected)}")
    if args.seed is not None:
        print(f"Random seed: {args.seed}")

    sources = [source for _, source in selected]
    try:
        outputs = {
            "statement_info": load_rows_for_sources(out_dir / "statement_info.csv", sources),
            "transactions": load_rows_for_sources(out_dir / "transactions.csv", sources),
        }
    except Exception as exc:
        print(f"ERROR reading CSV outputs: {type(exc).__name__}: {exc}")
        return 2

    results = [
        compare_statement(path, source, outputs, manifest_latest)
        for path, source in selected
    ]
    passed_count = sum(results)
    failed_count = len(results) - passed_count
    print("\n=== SUMMARY ===")
    print(f"Passed: {passed_count}; failed: {failed_count}; checked: {len(results)}")
    print("NOTE: this verifies saved-output consistency with the current parser, not independent extraction accuracy.")
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
