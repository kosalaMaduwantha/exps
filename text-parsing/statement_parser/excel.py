"""Excel workbook input/output for parsed statements.

Single-file mode writes one Statement_Info row; batch mode writes one row per
parsed file. Incremental runs read the previous workbook back, merge new rows
into it, and rewrite the whole file (openpyxl in-place appending is not
reliable, and rewriting also reapplies the styling).

Note: the rewrite keeps our three sheets (Transactions, Statement_Info,
Errors). Any extra sheets a user added manually are not preserved.
"""

import re
from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter

from .patterns import OTHER_FIELD_PREFIX

# Matches placeholder column names such as "header_17".
_PLACEHOLDER_COLUMN = re.compile(rf"{re.escape(OTHER_FIELD_PREFIX)}(\d+)")


def style_sheet(ws, money_cols=()):
    font = Font(name="Arial", size=10)
    bold = Font(name="Arial", size=10, bold=True)
    for row in ws.iter_rows():
        for c in row:
            c.font = font
    for c in ws[1]:
        c.font = bold
        c.alignment = Alignment(horizontal="center")
    for col_idx, col in enumerate(ws.columns, start=1):
        # default=0 keeps empty sheets (no data rows) from raising max()
        width = max(
            (len(str(c.value)) if c.value is not None else 0 for c in col),
            default=0,
        )
        ws.column_dimensions[get_column_letter(col_idx)].width = min(
            max(width + 2, 10), 60
        )
    headers = [c.value for c in ws[1]]
    for name in money_cols:
        if name in headers:
            col = headers.index(name) + 1
            for r in range(2, ws.max_row + 1):
                ws.cell(row=r, column=col).number_format = (
                    "#,##0.00;(#,##0.00);-"
                )
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions


def _as_frame(obj):
    """Normalize list-of-dicts / single dict / DataFrame into a DataFrame."""
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, dict):
        obj = [obj]
    return pd.DataFrame(obj)


def _money_cols(df):
    """Column names that are numeric anywhere in the frame."""
    if df.empty:
        return ()
    return tuple(
        str(c) for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
    )


def money_columns(rows):
    """Column names (in first-seen order) whose value is numeric somewhere."""
    names = []
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)) and key not in names:
                names.append(key)
    return tuple(names)


def errors_frame(errors):
    """Build the Errors sheet frame from (Source_File, message) tuples."""
    if isinstance(errors, pd.DataFrame):
        return errors
    return pd.DataFrame(
        [{"Source_File": rel, "Error": msg} for rel, msg in errors],
        columns=["Source_File", "Error"],
    )


def read_workbook(path):
    """Load an existing output workbook as {sheet_name: DataFrame}.

    Returns None when the file does not exist. Raises on unreadable/corrupt
    files so the caller can decide what to do (e.g. suggest --fresh).
    """
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_excel(path, sheet_name=None)


def existing_columns(workbook):
    """Ordered column names across the sheets that carry parsed data."""
    columns = []
    for name in ("Statement_Info", "Transactions", "Errors"):
        df = (workbook or {}).get(name)
        if df is not None:
            for col in df.columns:
                if col not in columns:
                    columns.append(col)
    return columns


def processed_files(workbook):
    """Source_File values already recorded in a previously written workbook.

    This is the registry that makes incremental runs work: a file whose
    relative path appears here is skipped on the next run.
    """
    processed = set()
    for df in (workbook or {}).values():
        if df is not None and "Source_File" in df.columns:
            processed.update(df["Source_File"].dropna().astype(str))
    return processed


def placeholder_start(workbook):
    """First free header_N index for an existing workbook (1 when absent).

    Incremental runs must CONTINUE the numbering: reusing header_1 for a new
    file would pour an unrelated value into an existing column.
    """
    max_seen = 0
    for df in (workbook or {}).values():
        if df is None:
            continue
        for col in df.columns:
            m = _PLACEHOLDER_COLUMN.fullmatch(str(col))
            if m:
                max_seen = max(max_seen, int(m.group(1)))
    return max_seen + 1


def merge_frames(old_df, new_rows):
    """Append new row dicts to an existing DataFrame.

    Existing columns keep their order; columns first seen in `new_rows` are
    appended at the end. Empty inputs are handled on either side.
    """
    new_df = pd.DataFrame(new_rows)
    if old_df is None:
        return new_df
    if new_df.empty and len(new_df.columns) == 0:
        return old_df
    if len(old_df) == 0 and len(old_df.columns) == 0:
        return new_df
    return pd.concat([old_df, new_df], ignore_index=True)


def merge_errors(old_df, new_errors, attempted):
    """Merge the Errors sheet across incremental runs.

    Old error rows for files that were attempted again are dropped first, so
    a retried file ends up with exactly one current row (or none if it now
    parses cleanly).
    """
    if old_df is None or "Source_File" not in getattr(old_df, "columns", []):
        base = pd.DataFrame()
    else:
        base = old_df[~old_df["Source_File"].isin(attempted)]
    rows = [{"Source_File": rel, "Error": msg} for rel, msg in new_errors]
    return merge_frames(base, rows)


def write_workbook(out, transactions, statement_info, errors=None):
    """Write the Transactions / Statement_Info (and Errors) sheets.

    `transactions` and `statement_info` may be a DataFrame, a list of row
    dicts, or a single row dict. `errors` may be None (no Errors sheet at
    all), a list of (Source_File, message) tuples, or a DataFrame; an empty
    error list writes no sheet so clean runs keep the two-sheet layout.
    """
    df_txn = _as_frame(transactions)
    df_info = _as_frame(statement_info)

    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        df_txn.to_excel(xw, sheet_name="Transactions", index=False)
        df_info.to_excel(xw, sheet_name="Statement_Info", index=False)
        style_sheet(xw.sheets["Transactions"], money_cols=("Amount",))
        style_sheet(xw.sheets["Statement_Info"], money_cols=_money_cols(df_info))
        if errors is not None:
            df_err = errors_frame(errors)
            if len(df_err) > 0:
                df_err.to_excel(xw, sheet_name="Errors", index=False)
                style_sheet(xw.sheets["Errors"])

    return df_txn, df_info
