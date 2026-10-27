"""Streaming CSV outputs, file manifest and run state for parsed statements.

Replaces the old single-workbook xlsx writer. Everything here is stdlib-only
(no pandas/openpyxl), so the extractor runs on a bare Python install.

Output directory layout:
    transactions.csv     one row per transaction (append-only)
    statement_info.csv   one row per parsed file (append-only)
    other_fields.csv     optional long format (Source_File, Line_No, Column_Name,
                         Value); written only with --include-other
    errors.csv           regenerated from the manifest after every run; present
                         only when errors exist
    manifest.jsonl       one JSON record per processed file (append-only; the
                         last record per source wins)
    state.json           naming state for --include-other; rewritten atomically

Incremental behavior: a source whose latest manifest record has status "ok"
is skipped on later runs; any other source ("error") is retried, so rerunning
after fixing bad files reprocesses exactly those files.

Crash safety: every manifest record carries the cumulative CSV byte offsets
as of the batch it belongs to (rows are flushed + fsynced BEFORE the records
are appended). On the next run, CSVs longer than the last valid manifest
checkpoint are truncated back, and files without an "ok" record are parsed
again, so an interrupted batch can never leave duplicate or orphan rows.
Torn manifest tail lines (invalid JSON from an interrupted append) are
ignored and their bytes recovered the same way.
"""

import csv
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from .patterns import OTHER_FIELD_PREFIX
from .parsing import parse_statement

TXN_COLUMNS = (
    "Source_File", "Card_Number", "Owner_Key", "Post_Date",
    "Transaction_Date", "Description", "Transaction_Type",
    "Amount", "Amount_Flag", "Source_Line",
)
INFO_COLUMNS = (
    "Source_File", "Customer_Name", "Customer_Address", "Card_Number",
    "Statement_Date", "Total_Outstanding", "Credit_Limit",
    "Payment_Due_Date", "Minimum_Payment", "Opening_Balance", "Debit",
    "Credit", "Closing_Balance", "Total_Debit",
    "Sub_Total_Debit_01", "Sub_Total_Debit_02",
)
OTHER_COLUMNS = ("Source_File", "Line_No", "Column_Name", "Value")
ERROR_COLUMNS = ("Source_File", "Error")

TXN_OFFSET_KEY = "txn_offset"
INFO_OFFSET_KEY = "info_offset"
OTHER_OFFSET_KEY = "other_offset"
OFFSET_KEYS = (TXN_OFFSET_KEY, INFO_OFFSET_KEY, OTHER_OFFSET_KEY)

# header fields read_header() must provide; a mismatch fails loudly instead
# of silently dropping columns if the parser's header schema ever changes
REQUIRED_INFO_FIELDS = frozenset(INFO_COLUMNS) - {"Source_File"}

OUTPUT_FILES = (
    "transactions.csv", "statement_info.csv", "other_fields.csv",
    "errors.csv", "manifest.jsonl", "state.json",
)

# a batch is also flushed when it accumulates this many transaction rows
ROW_FLUSH_LIMIT = 500_000

_PLACEHOLDER_COLUMN = re.compile(rf"{re.escape(OTHER_FIELD_PREFIX)}(\d+)")


def _cell(value):
    return "" if value is None else value


def txn_row(source, txn):
    """List-form transaction row in TXN_COLUMNS order."""
    return [source] + [_cell(txn.get(c, "")) for c in TXN_COLUMNS[1:]]


def info_row(source, header):
    """List-form statement-info row in INFO_COLUMNS order."""
    missing = REQUIRED_INFO_FIELDS - set(header)
    if missing:
        raise ValueError(
            "parser header is missing required field(s): "
            + ", ".join(sorted(missing))
        )
    return [source] + [_cell(header.get(c, "")) for c in INFO_COLUMNS[1:]]


def parse_file_task(payload):
    """Worker: parse one statement file (pure; safe in a process pool).

    payload: (root, rel, include_other)
    result:  (rel, ok, header, transactions, others, header_lines, message)
             `others`/`header_lines` are dropped when include_other is False
             to keep inter-process traffic small.
    """
    root, rel, include_other = payload
    path = Path(root) / rel
    try:
        header, txns, others, header_lines = parse_statement(path)
    except Exception as exc:
        return (rel, False, None, None, None, None, f"{type(exc).__name__}: {exc}")
    if not header and not txns and not others:
        return (rel, False, None, None, None, None, "no data found")
    if not include_other:
        others, header_lines = [], set()
    return (rel, True, header, txns, others, header_lines, None)


def placeholder_next(names):
    """First free header_N index for a set of column names (1 when absent)."""
    max_seen = 0
    for name in names:
        m = _PLACEHOLDER_COLUMN.fullmatch(str(name))
        if m:
            max_seen = max(max_seen, int(m.group(1)))
    return max_seen + 1


class Manifest:
    """Append-only JSON-lines registry; the last record per source wins."""

    def __init__(self, path):
        self.path = Path(path)
        self._file = None

    def load_latest(self):
        return load_checkpoint(self.path)[0]

    def append(self, records):
        if self._file is None:
            self._file = self.path.open("a", encoding="utf-8")
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for record in records:
            record = {"ts": ts, **record}
            self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()
        os.fsync(self._file.fileno())

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None


def load_checkpoint(path):
    """Read the whole manifest.

    Returns (latest, offsets):
      latest  - {source: record}, the LAST record per source wins
      offsets - cumulative CSV byte offsets from the last valid record
    Lines that are not valid JSON (a torn append after a crash) are ignored;
    their rows are recovered via the offsets of the last valid record.
    """
    path = Path(path)
    latest, offsets = {}, {key: 0 for key in OFFSET_KEYS}
    if not path.exists():
        return latest, offsets
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            latest[record.get("source")] = record
            for key in OFFSET_KEYS:
                value = record.get(key)
                if value is not None:
                    offsets[key] = value
    return latest, offsets


class State:
    """Naming state for --include-other; rewritten atomically per batch."""

    DEFAULTS = {
        "version": 1,
        "placeholder_counter": 1,
        "known_columns": [],
    }

    def __init__(self, path):
        self.path = Path(path)
        self.data = dict(self.DEFAULTS)
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                self.data.update(json.load(f))

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def save(self):
        tmp = self.path.with_name(self.path.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)


class CsvStore:
    """Streaming append-only writer for one CSV with a fixed header row."""

    def __init__(self, path, columns):
        self.path = Path(path)
        self.columns = columns
        self._file = None
        self._writer = None

    def open(self):
        fresh = not (self.path.exists() and self.path.stat().st_size > 0)
        self._file = self.path.open("a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        if fresh:
            self._writer.writerow(self.columns)
            self._file.flush()
        return self

    def append_rows(self, rows):
        if rows:
            self._writer.writerows(rows)

    def checkpoint(self):
        """Flush to disk and return the current on-disk size in bytes."""
        self._file.flush()
        os.fsync(self._file.fileno())
        return os.fstat(self._file.fileno()).st_size

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None


def recover_outputs(out, offsets):
    """Truncate CSVs back to the last manifest checkpoint (crash recovery).

    Must run BEFORE the stores are opened, so a fresh header row is not
    mistaken for uncheckpointed data.
    """
    names = {
        "transactions.csv": "txn_offset",
        "statement_info.csv": "info_offset",
        "other_fields.csv": "other_offset",
    }
    for name, key in names.items():
        path = Path(out) / name
        offset = offsets[key]
        if not path.exists():
            if offset:
                raise RuntimeError(
                    f"{name} is missing but the manifest checkpoint says "
                    f"{offset} bytes; rerun with --fresh."
                )
            continue
        size = path.stat().st_size
        if size > offset:
            print(f"Recovering {name}: dropping {size - offset} uncheckpointed "
                  f"byte(s) from an interrupted run")
            with path.open("r+b") as f:
                f.truncate(offset)
                f.flush()
                os.fsync(f.fileno())
        elif size < offset:
            raise RuntimeError(
                f"{name} is shorter ({size} bytes) than the manifest "
                f"checkpoint ({offset} bytes); rerun with --fresh."
            )


def write_errors_csv(out, latest):
    """Regenerate errors.csv from the manifest's current error records."""
    path = Path(out) / "errors.csv"
    errors = sorted(
        (record["source"], record.get("error") or "")
        for record in latest.values()
        if record.get("status") != "ok"
    )
    if not errors:
        path.unlink(missing_ok=True)
        return 0
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(ERROR_COLUMNS)
        writer.writerows(errors)
    return len(errors)


def reset_outputs(out):
    """Delete every artifact this package owns inside the output directory."""
    out = Path(out)
    for part in out.glob("*_part_*.csv"):
        part.unlink()
    for name in OUTPUT_FILES:
        (out / name).unlink(missing_ok=True)


def count_data_rows(path):
    """Number of data rows (excluding the header) of a CSV file."""
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        return sum(1 for _ in reader)


def split_csv(path, part_rows, pad=3):
    """Split a CSV into part files of at most part_rows data rows each.

    Every part repeats the header row, so each part is a standalone CSV that
    also opens directly in Excel (which silently drops rows past 1,048,576
    when a large CSV is opened). Returns the list of part paths.
    """
    path = Path(path)
    part_rows = max(1, part_rows)
    parts = []
    with path.open("r", newline="", encoding="utf-8") as f:
        header = next(csv.reader(f), None)
        if header is None:
            return parts
        index, rows = 0, []
        for row in csv.reader(f):
            rows.append(row)
            if len(rows) >= part_rows:
                index += 1
                parts.append(_write_part(path, index, header, rows, pad))
                rows = []
        if rows:
            index += 1
            parts.append(_write_part(path, index, header, rows, pad))
    return parts


def _write_part(path, index, header, rows, pad):
    part = path.with_name(f"{path.stem}_part_{index:0{pad}d}{path.suffix}")
    with part.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return part


def collect_stats(out):
    """Summary of an output directory from its manifest and file sizes."""
    out = Path(out)
    latest = load_checkpoint(out / "manifest.jsonl")[0]
    ok = [r for r in latest.values() if r.get("status") == "ok"]
    errors = [r for r in latest.values() if r.get("status") != "ok"]
    return {
        "ok": len(ok),
        "errors": len(errors),
        "txn_rows": sum(r.get("txn", 0) for r in ok),
        "info_rows": len(ok),
        "error_messages": [r.get("error") or "" for r in errors],
        "files": [
            (p.name, p.stat().st_size)
            for p in sorted(out.iterdir())
            if p.is_file()
        ],
    }
