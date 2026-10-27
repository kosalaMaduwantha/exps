# Statement Parser — CSV Pipeline Execution Guide

The pipeline parses raw credit-card statement `.txt` files and writes streaming
CSV outputs (no Excel/xlsx — CSVs don't have Excel's 1,048,576-row truncation
limit, and the writer needs only the Python standard library).

Source of truth for this pipeline: `~/git-repos/text_parsing_scripting/`
(modules: `data_extractor.py` + `statement_parser/{parsing,csv_io}.py`).
The `exps/text-parsing/` directory keeps the earlier JSON prototypes.

## Quick start

```bash
cd ~/git-repos/text_parsing_scripting
./env/bin/python data_extractor.py ./statements ./out
```

- `src` — one statement `.txt` file, or a directory (scanned recursively for
  `*.txt`, use `--pattern` to change the glob)
- `out` — output directory (required for directory mode)

Outputs (CSV, UTF-8, RFC-4180 quoted):

| File | Contents |
|---|---|
| `transactions.csv` | one row per transaction |
| `statement_info.csv` | one row per statement + statement fields |
| `other_fields.csv` | (only with `--include-other`) non-transaction lines as long-format rows |
| `manifest.jsonl` | per-file record: status, row counts, CSV byte offsets |
| `errors.csv` | (only when errors exist) files that failed with a reason |

## Common operations

```bash
# Re-run later: only new (and previously-errored) files are parsed; existing
# output is appended to exactly once.
./env/bin/python data_extractor.py ./statements ./out

# Summary of an existing output directory
./env/bin/python data_extractor.py ./out --stats

# Files currently flagged as errors (fix them, then re-run to retry)
./env/bin/python data_extractor.py ./out --list-errors

# Split CSVs into ~1,000,000-row parts (each part opens fully in Excel)
./env/bin/python data_extractor.py ./out --split 1000000

# Ignore previous outputs and parse everything from scratch
./env/bin/python data_extractor.py ./statements ./out --fresh

# Also capture non-transaction lines
./env/bin/python data_extractor.py ./statements ./out --include-other

# Tuning (defaults: all cores, flush every 10 000 files)
./env/bin/python data_extractor.py ./statements ./out --workers 8 --batch-size 10000 --verbose
```

Exit codes: `0` clean, `1` some files errored (retry after fixing), `2` bad usage.

## Performance

Measured on 100k files / 217 MB (8 workers): ~66 s, bounded peak RAM (~0.5 GB).
Extrapolated 1 GB corpus (~460k files): ~5–6 minutes. Re-running an
already-complete corpus is I/O-bound discovery only (~40 s / 100k files).

Do not open the full `transactions.csv` directly in Excel (silent truncation);
use `--split` parts, `--stats`, or the validator.

## Validation

```bash
python test/validator.py <src> <out> [--samples N] [--seed S]
```

Re-parses a sample of sources and compares them to the saved CSV rows.
(Note: PDF/HTML-derived sources are inherently non-reproducible; those are
reported as uncheckable rather than failures.)
