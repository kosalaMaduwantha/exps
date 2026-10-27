"""Parse bank credit statement text files into streaming CSV outputs.

Usage:
    python data_extractor.py sample_statement.txt [out_dir]
    python data_extractor.py ./statements [out_dir] [--pattern "*.txt"] [flags]

Directory mode recursively parses every file matching --pattern and appends
rows to the CSV files inside the output directory:

    transactions.csv     one row per transaction from every parsed file
    statement_info.csv   one row per parsed file (wide header fields)
    other_fields.csv     only with --include-other: non-transaction lines as
                         long-format rows (Source_File, Line_No, Column_Name,
                         Value)
    errors.csv           current error list (regenerated from the manifest)

Re-running against the same parent directory is INCREMENTAL: files whose
latest manifest.jsonl record has status "ok" are skipped. Files flagged
"error" are retried automatically on the next run, so fixing a bad file and
rerunning reprocesses exactly that file. Use --fresh to ignore previous
outputs and parse everything again.

Parallelism / memory:
    Parsing runs in a process pool (--workers, default: all CPU cores) and
    rows are streamed to the CSVs in batches (--batch-size files), so peak
    memory stays bounded regardless of input size. Output row order is the
    sorted file order and identical for any worker count.

Query / post-process modes (no parsing):
    --stats              summary of an existing output directory
    --list-errors       files currently flagged as errors in the manifest
    --split N           split transactions.csv (and statement_info.csv when
                        larger) into parts of N data rows; each part repeats
                        the header and opens directly in Excel

This entrypoint only orchestrates: discover -> parse -> stream to CSV.
All parsing logic lives in statement_parser.parsing (unchanged).
"""

import argparse
import itertools
import multiprocessing
import sys
import time
from collections import Counter
from pathlib import Path

from statement_parser.csv_io import (
    INFO_COLUMNS,
    INFO_OFFSET_KEY,
    OTHER_COLUMNS,
    OTHER_OFFSET_KEY,
    ROW_FLUSH_LIMIT,
    TXN_COLUMNS,
    TXN_OFFSET_KEY,
    CsvStore,
    Manifest,
    State,
    collect_stats,
    count_data_rows,
    info_row,
    load_checkpoint,
    parse_file_task,
    placeholder_next,
    reset_outputs,
    recover_outputs,
    split_csv,
    txn_row,
    write_errors_csv,
)
from statement_parser.parsing import (
    BASE_FIELD_NAMES,
    discover_statement_files,
    pieces_to_columns,
    split_other_line,
)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Parse statement text file(s) into streaming CSV outputs.",
    )
    parser.add_argument("src", nargs="?", default=None,
                        help="statement .txt file, a parent directory, or the "
                             "output directory for --stats/--list-errors/--split")
    parser.add_argument("out", nargs="?", default=None,
                        help="output directory (optional)")
    parser.add_argument("--pattern", default="*.txt",
                        help="file glob for directory mode (default: %(default)s)")
    parser.add_argument("--fresh", action="store_true",
                        help="ignore previous outputs and parse everything")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count(),
                        help="parser processes for directory mode (default: all cores)")
    parser.add_argument("--batch-size", type=int, default=10_000,
                        help="files per CSV/manifest flush (default: %(default)s)")
    parser.add_argument("--include-other", action="store_true",
                        help="also write non-transaction lines as long-format "
                             "rows to other_fields.csv")
    parser.add_argument("--split", type=int, metavar="N", default=None,
                        help="after parsing, split CSVs into parts of N data "
                             "rows (each part opens in Excel)")
    parser.add_argument("--stats", action="store_true",
                        help="print a summary of an existing output directory")
    parser.add_argument("--list-errors", action="store_true",
                        help="print files currently flagged as errors")
    parser.add_argument("--verbose", action="store_true",
                        help="print one line per parsed file")
    return parser


def resolve_paths(args):
    """Resolve the input path, the output directory, and query-mode paths."""
    if args.src is None:
        raise SystemExit(
            "pass a statement file / parent directory (or the output "
            "directory when using --stats / --list-errors / --split)"
        )
    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"input not found: {src}")

    query = args.stats or args.list_errors or args.split is not None
    src_is_output = (
        args.out is None and query and src.is_dir()
        and (src / "manifest.jsonl").exists()
    )
    if args.out:
        out = Path(args.out)
    elif src_is_output:
        out = src
    elif src.is_dir():
        out = src.with_name(f"{src.name}_parsed_csv")
    else:
        out = src.parent / f"{src.stem}_parsed_csv"
    return (None, out) if src_is_output else (src, out)


class Runner:
    """Bounded-memory batch runner: consume parse results, stream to CSVs.

    Batch ordering (crash safety): CSV rows -> flush+fsync (byte offsets) ->
    manifest records carrying those offsets -> naming state. An interrupted
    batch is detected on the next run via recover_outputs() and reprocessed.
    """

    def __init__(self, args, out):
        self.out = Path(out)
        self.args = args
        self.verbose = args.verbose
        self.include_other = args.include_other
        self.txn = CsvStore(self.out / "transactions.csv", TXN_COLUMNS).open()
        self.info = CsvStore(self.out / "statement_info.csv", INFO_COLUMNS).open()
        self.other = (
            CsvStore(self.out / "other_fields.csv", OTHER_COLUMNS).open()
            if self.include_other else None
        )
        self.manifest = Manifest(self.out / "manifest.jsonl")
        self.state = State(self.out / "state.json")
        self.counter = itertools.count(self.state["placeholder_counter"])
        self.used = set(self.state["known_columns"]) | set(BASE_FIELD_NAMES)
        self.batch_txn, self.batch_info = [], []
        self.batch_other, self.batch_records = [], []
        self.n_ok = self.n_err = 0
        self.started = time.monotonic()

    def handle(self, result):
        rel, ok, header, txns, others, header_lines, message = result
        if not ok:
            self.n_err += 1
            self.batch_records.append({
                "source": rel, "status": "error",
                "txn": 0, "info": 0, "error": message,
            })
            if self.verbose:
                print(f"  ERROR {rel}: {message}")
        else:
            self.n_ok += 1
            self.batch_txn.extend(txn_row(rel, t) for t in txns)
            self.batch_info.append(info_row(rel, header))
            self.batch_records.append({
                "source": rel, "status": "ok",
                "txn": len(txns), "info": 1, "error": None,
            })
            if others:
                self._other_rows(rel, header, others, header_lines)
            if self.verbose:
                print(f"  {rel}: {len(txns)} transactions")
        if (len(self.batch_records) >= self.args.batch_size
                or len(self.batch_txn) >= ROW_FLUSH_LIMIT):
            self.flush()

    def _other_rows(self, rel, header, others, header_lines):
        """Long-format rows for non-transaction lines.

        Uses the same naming machinery as build_statement_info (same
        counter, same `used` namespace, per-file `file_used`), but emits one
        row per named piece so line numbers are kept and the file stays
        append-only regardless of discovered labels.
        """
        self.used.update(header)
        file_used = set(header)
        for lineno, line in others:
            if lineno in header_lines:
                continue
            for name, value in pieces_to_columns(
                split_other_line(line), self.counter, self.used, file_used
            ):
                self.batch_other.append([rel, lineno, name, value])

    def flush(self):
        if not (self.batch_records or self.batch_txn or self.batch_info
                or self.batch_other):
            return
        self.txn.append_rows(self.batch_txn)
        self.info.append_rows(self.batch_info)
        if self.other is not None and self.batch_other:
            self.other.append_rows(self.batch_other)
        txn_off = self.txn.checkpoint()
        info_off = self.info.checkpoint()
        other_off = self.other.checkpoint() if self.other is not None else 0
        records = [
            {**record,
             TXN_OFFSET_KEY: txn_off,
             INFO_OFFSET_KEY: info_off,
             OTHER_OFFSET_KEY: other_off}
            for record in self.batch_records
        ]
        self.manifest.append(records)
        if self.other is not None:
            self.state["placeholder_counter"] = placeholder_next(self.used)
            self.state["known_columns"] = sorted(self.used)
            self.state.save()
        self.batch_txn.clear()
        self.batch_info.clear()
        self.batch_other.clear()
        self.batch_records.clear()

    def finish(self):
        """Flush the remainder and report the run summary."""
        self.flush()
        latest = load_checkpoint(self.manifest.path)[0]
        n_flagged = write_errors_csv(self.out, latest)
        ok_records = [r for r in latest.values() if r.get("status") == "ok"]
        total_txn = sum(r.get("txn", 0) for r in ok_records)
        elapsed = time.monotonic() - self.started
        rate = (self.n_ok + self.n_err) / elapsed if elapsed else 0.0
        print(f"Parsed {self.n_ok} file(s), {self.n_err} error(s) | "
              f"{total_txn:,} total transaction rows, "
              f"{len(ok_records):,} info rows -> {self.out}")
        print(f"Elapsed {elapsed:.1f}s ({rate:.0f} files/s, "
              f"{self.args.workers} worker(s))")
        if n_flagged:
            print(f"{n_flagged} file(s) flagged as errors in the manifest - "
                  f"inspect with: python data_extractor.py {self.out} "
                  f"--list-errors")

    def close(self):
        self.txn.close()
        self.info.close()
        if self.other is not None:
            self.other.close()
        self.manifest.close()


def run_pipeline(args, src, out):
    """Parse one statement file or a whole directory into the CSV outputs."""
    out = Path(out)
    if out.exists() and not out.is_dir():
        raise SystemExit(f"output path exists and is not a directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    if args.fresh:
        reset_outputs(out)
        print(f"Fresh run: previous outputs in {out} were reset")

    latest, offsets = load_checkpoint(out / "manifest.jsonl")
    has_data = any(
        (out / name).stat().st_size > 0
        for name in ("transactions.csv", "statement_info.csv",
                     "other_fields.csv")
        if (out / name).exists()
    )
    if has_data and not latest:
        raise SystemExit(
            f"{out} contains CSV data but the manifest has no records "
            f"(was it deleted?); use --fresh or remove the directory"
        )
    recover_outputs(out, offsets)

    runner = Runner(args, out)
    try:
        if src.is_file():
            rc = run_single(args, src, runner, latest)
        else:
            rc = run_directory(args, src, runner, latest)
        runner.finish()
        return rc
    finally:
        runner.close()


def run_single(args, src, runner, latest):
    """Parse one statement file and append it to the CSV outputs."""
    rel = src.name
    record = latest.get(rel)
    if record is not None and record.get("status") == "ok":
        print(f"Already processed: {rel} - nothing to do")
        return 0
    runner.handle(parse_file_task(
        (str(src.parent), src.name, runner.include_other)
    ))
    return 0 if runner.n_ok else 1


def run_directory(args, src, runner, latest):
    """Recursively parse new files under src and stream them to the CSVs."""
    files = discover_statement_files(src, args.pattern)
    out_resolved = runner.out.resolve()
    if src.resolve() in out_resolved.parents:
        # never discover files inside our own output directory
        files = [p for p in files
                 if out_resolved not in p.resolve().parents]
    entries = [(path, path.relative_to(src).as_posix()) for path in files]
    if not entries:
        print(f"No files matching pattern {args.pattern!r} under {src}")
        return 1
    skip = {s for s, r in latest.items() if r.get("status") == "ok"}
    todo = [(path, rel) for path, rel in entries if rel not in skip]
    if not todo:
        print("Nothing to do - every file is already processed.")
        return 0
    print(f"Found {len(entries)} file(s); {len(todo)} to process "
          f"({len(entries) - len(todo)} already ok) "
          f"[{args.workers} worker(s)]")
    run_parse_tasks(args, src, todo, runner)
    return 0 if runner.n_ok else 1


def run_parse_tasks(args, src, todo, runner):
    """Feed parse tasks to a pool (ordered results) or run them inline."""
    tasks = ((str(src), rel, runner.include_other) for _, rel in todo)
    workers = max(1, args.workers)
    if workers > 1 and len(todo) > 1:
        chunk = max(1, min(256, len(todo) // (workers * 16)))
        with multiprocessing.Pool(workers) as pool:
            for result in pool.imap(parse_file_task, tasks, chunk):
                runner.handle(result)
    else:
        for task in tasks:
            runner.handle(parse_file_task(task))


def run_query(args, out):
    """--stats / --list-errors: report on an existing output directory."""
    out = Path(out)
    if not (out / "manifest.jsonl").exists():
        raise SystemExit(
            f"no manifest found in {out} (is it an extractor output directory?)"
        )
    latest = load_checkpoint(out / "manifest.jsonl")[0]
    errors = sorted(
        (record["source"], record.get("error") or "")
        for record in latest.values()
        if record.get("status") != "ok"
    )
    if args.list_errors:
        if not errors:
            print("No files are currently flagged as errors.")
            return 0
        for source, message in errors:
            print(f"  {source}: {message}")
        print(f"{len(errors)} file(s) flagged; they are retried automatically "
              f"on the next run")
        return 1
    stats = collect_stats(out)
    print(f"Output directory : {out}")
    print(f"Files processed  : {stats['ok']} ok, {stats['errors']} error")
    print(f"Transactions     : {stats['txn_rows']:,} rows")
    print(f"Statement info   : {stats['info_rows']:,} rows")
    for name, size in stats["files"]:
        print(f"  {name:<18} {size:>12,} bytes")
    if errors:
        grouped = Counter(message[:80] for _, message in errors)
        print("Top error messages:")
        for message, count in grouped.most_common(5):
            print(f"  {count:>6} x {message}")
    return 0


def run_split(args, out):
    """--split N: cut the CSVs into Excel-friendly part files."""
    out = Path(out)
    if not out.is_dir():
        raise SystemExit(f"output directory not found: {out}")
    for name in ("transactions.csv", "statement_info.csv"):
        path = out / name
        if not path.exists():
            continue
        rows = count_data_rows(path)
        if rows <= args.split:
            print(f"{name}: {rows:,} data rows -> not split "
                  f"(limit {args.split:,})")
            continue
        parts = split_csv(path, args.split)
        print(f"{name}: {rows:,} data rows -> {len(parts)} part(s) of "
              f"<= {args.split:,} rows")
        for part in parts:
            print(f"  {part.name} ({part.stat().st_size:,} bytes)")
    return 0


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    src, out = resolve_paths(args)
    try:
        if args.stats or args.list_errors:
            return run_query(args, out)
        if src is not None:
            rc = run_pipeline(args, src, out)
            if rc != 0:
                return rc
        if args.split is not None:
            return run_split(args, out)
        return 0
    except OSError as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
