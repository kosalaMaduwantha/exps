"""
Parse bank credit statement text files into one Excel workbook.

Usage:
    python parse_to_json_v3.py sample_statement.txt [out.xlsx]
    python parse_to_json_v3.py ./statements [out.xlsx] [--pattern "*.txt"] [--fresh]

Directory mode recursively parses every file matching --pattern under the
parent directory and appends the results into ONE workbook:

    Transactions   - one row per transaction from every parsed file
    Statement_Info - one row per parsed file (wide layout, see parsing.py)
    Errors         - files that could not be parsed (written only when
                     errors occurred)

Re-running against the same parent directory is INCREMENTAL: files already
recorded (matched by their relative path in the Source_File column) are
skipped, so only new files are parsed and appended. Placeholder columns
continue numbering after the existing workbook. Use --fresh to ignore the
existing workbook and parse everything again.

This entrypoint only orchestrates: discover/parse -> merge -> write Excel.
"""

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

from statement_parser.excel import (
    errors_frame,
    existing_columns,
    merge_errors,
    merge_frames,
    placeholder_start,
    processed_files,
    read_workbook,
    write_workbook,
)
from statement_parser.parsing import (
    build_statement_info,
    parse_directory,
    parse_statement,
)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Parse statement text file(s) into an Excel workbook.",
    )
    parser.add_argument("src", help="statement .txt file or a parent directory")
    parser.add_argument("out", nargs="?", default=None,
                        help="output .xlsx path (optional)")
    parser.add_argument("--pattern", default="*.txt",
                        help="file glob for directory mode (default: %(default)s)")
    parser.add_argument("--fresh", action="store_true",
                        help="ignore an existing output workbook and parse everything")
    return parser


def resolve_paths(args):
    """Resolve the input path and the default output path."""
    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"input not found: {src}")
    if args.out:
        out = Path(args.out)
    elif src.is_dir():
        out = src.with_name(f"{src.name}_parsed.xlsx")
    else:
        out = src.with_name(f"{src.stem}_v3.xlsx")
    return src, out


def load_existing_workbook(out, fresh):
    """Load the previous output for an incremental run (None when fresh).

    Aborts with guidance when the workbook exists but cannot be used as a
    registry (unreadable, or written before Source_File tracking existed).
    """
    if fresh or not out.exists():
        return None
    try:
        workbook = read_workbook(out)
    except Exception as exc:
        raise SystemExit(
            f"Could not read existing workbook {out}: {exc}\n"
            f"Use --fresh to ignore it or choose a different output name."
        )
    tracked = any(
        "Source_File" in getattr(df, "columns", [])
        for df in workbook.values()
    )
    if not tracked:
        raise SystemExit(
            f"{out} has no Source_File tracking; rerun with --fresh or "
            f"choose a new output name."
        )
    return workbook


def run_single(src, out, workbook, fresh):
    """Parse one statement file and write/merge it into the workbook."""
    if not fresh and workbook is not None:
        if src.name in processed_files(workbook):
            print(f"Already processed: {src.name} - nothing to do")
            return 0

    header, txns, others, header_lines = parse_statement(src)
    if not header and not txns and not others:
        print(f"ERROR {src.name}: no data found")
        return 1

    # Continue placeholder numbering / label reuse from the existing workbook.
    known = existing_columns(workbook) if workbook else []
    start = placeholder_start(workbook) if workbook else 1
    info = build_statement_info(
        header, others, header_lines,
        counter=itertools.count(start),
        used=set(known),
        file_used=set(header),
    )

    rel = src.name
    new_txn = [{"Source_File": rel, **t} for t in txns]
    new_info = [{"Source_File": rel, **info}]

    if workbook is None:
        df_txn = pd.DataFrame(new_txn)
        df_info = pd.DataFrame(new_info)
        df_err = errors_frame([])
    else:
        df_txn = merge_frames(workbook.get("Transactions"), new_txn)
        df_info = merge_frames(workbook.get("Statement_Info"), new_info)
        df_err = merge_errors(workbook.get("Errors"), [], {rel})

    write_workbook(out, df_txn, df_info, errors=df_err)
    print(f"Parsed {len(df_txn)} transactions and "
          f"{len(df_info.columns)} statement columns -> {out}")
    return 0


def run_directory(src, out, workbook, args):
    """Recursively parse new files under src and merge them into the workbook."""
    if workbook is None:
        result = parse_directory(src, args.pattern, exclude=out)
    else:
        result = parse_directory(
            src,
            args.pattern,
            skip_files=processed_files(workbook),
            counter_start=placeholder_start(workbook),
            known_columns=existing_columns(workbook),
            exclude=out,
        )

    if result["found"] == 0:
        print(f"No files matching pattern {args.pattern!r} under {src}")
        return 1
    if not result["parsed"] and not result["errors"]:
        print("Nothing to do - every file is already processed.")
        return 0

    for rel, n_txn, n_fields in result["parsed"]:
        print(f"  {rel}: {n_txn} transactions, {n_fields} statement fields")
    for rel, message in result["errors"]:
        print(f"  ERROR {rel}: {message}")

    attempted = ({rel for rel, _, _ in result["parsed"]}
                 | {rel for rel, _ in result["errors"]})
    if workbook is None:
        df_txn = pd.DataFrame(result["transactions"])
        df_info = pd.DataFrame(result["info_rows"])
        df_err = errors_frame(result["errors"])
    else:
        df_txn = merge_frames(workbook.get("Transactions"), result["transactions"])
        df_info = merge_frames(workbook.get("Statement_Info"), result["info_rows"])
        df_err = merge_errors(workbook.get("Errors"), result["errors"], attempted)

    write_workbook(out, df_txn, df_info, errors=df_err)
    print(f"Parsed {len(result['parsed'])} file(s), "
          f"{len(df_txn)} total transactions -> {out}")
    return 0 if result["parsed"] else 1


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    src, out = resolve_paths(args)
    workbook = load_existing_workbook(out, args.fresh)
    try:
        if src.is_dir():
            return run_directory(src, out, workbook, args)
        return run_single(src, out, workbook, args.fresh)
    except OSError as exc:
        print(f"ERROR writing {out}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
