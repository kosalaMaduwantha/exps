"""
Parse a (text-based) bank credit statement into an Excel workbook.

Usage:
    python parse_to_json_v3.py sample_statement.txt sample_statement_output_v3.xlsx

Requires: pandas, openpyxl   (pip install pandas openpyxl)

Output sheets
    Transactions   - one row per transaction, each tagged with the statement owner
    Statement_Info - ONE row, wide layout. Owner/account details plus every other
                     piece of data found in the statement. Columns that have a
                     meaningful label in the source line use that label
                     (e.g. "TOTAL - DEBITS"); unlabeled values fall back to
                     header_1, header_2, ...

This entrypoint only orchestrates: parse -> build statement info -> write Excel.
"""

import sys
from pathlib import Path

from statement_parser import build_statement_info, parse_statement, write_workbook


def resolve_paths(argv):
    if len(argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    src = Path(argv[1])
    out = (
        Path(argv[2])
        if len(argv) > 2
        else src.with_name(f"{src.stem}_v3.xlsx")
    )
    return src, out


def main(argv=None):
    argv = sys.argv if argv is None else argv
    src, out = resolve_paths(argv)

    header, txns, other_lines, header_lines = parse_statement(src)
    info = build_statement_info(header, other_lines, header_lines)
    df_txn, _ = write_workbook(out, txns, info)

    print(f"Parsed {len(df_txn)} transactions and {len(info)} statement fields -> {out}")
    if not df_txn.empty and df_txn["Amount"].isna().any():
        print("Warning: some transactions have no amount - check 'Source_Line' in the sheet.")


if __name__ == "__main__":
    main()
