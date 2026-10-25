# Text Parsing — bank statement → Excel

Parses (text-based) bank / credit-card statements into **one Excel workbook**:

| Sheet | Contents |
|---|---|
| `Transactions` | one row per transaction, from every parsed file |
| `Statement_Info` | one row per statement: owner/account header **plus every other piece of data** found in the file (balances, totals, sub totals), each value in its own column |
| `Errors` | files that could not be parsed (written only when errors occurred) |

Input is either a **single `.txt` file** or a **parent directory** scanned recursively.
Re-running against the same directory is **incremental**: files that were already
processed are skipped and only new files are appended.

`sample_statement.txt` is dummy data that follows the real statement layout
(replaced names, amounts and dates), so everything can be tested safely.

---

## Project layout

```
text-parsing/
├── parse_to_json_v3.py          # entrypoint: CLI, incremental logic, Excel hand-off
├── parse_to_json_v2.py          # previous single-file version (kept for reference)
├── parse_to_json_legacy.py      # original script (kept for reference)
├── sample_statement.txt         # dummy input
└── statement_parser/            # the actual library (functions only, no classes)
    ├── parsing.py               # ALL parsing logic (header, other fields, transactions, batch)
    ├── patterns.py              # every regex in one place
    ├── utils.py                 # to_number, clean, unique_name
    └── excel.py                 # workbook read / merge / write / styling
```

## Requirements & quick start

Python 3.12+ with `pandas` and `openpyxl`.

```bash
# single file
python parse_to_json_v3.py sample_statement.txt out.xlsx

# directory mode: recursively parses every *.txt under ./statements
python parse_to_json_v3.py ./statements                    # -> ./statements_parsed.xlsx
python parse_to_json_v3.py ./statements out.xlsx --pattern "*.txt"

# ignore an existing workbook and parse everything again
python parse_to_json_v3.py ./statements out.xlsx --fresh
```

Typical output:

```
  a.txt: 11 transactions, 20 statement fields
  sub1/b.txt: 11 transactions, 20 statement fields
  ERROR sub2/broken.txt: no data found
Parsed 4 file(s), 33 total transactions -> statements_parsed.xlsx
```

Exit codes: `0` = success (also when everything is already processed),
`1` = nothing parsed (empty directory / no data / write failure).

---

## Architecture

```
                        parse_to_json_v3.py  (entrypoint)
                 argparse · path resolution · incremental vs fresh
                                    │
                                    ▼
        ┌───────────────────────────────────────────────────────┐
        │                statement_parser.parsing               │
        │                                                       │
        │  read_header()           owner/account block          │
        │  parse_statement()       classify lines, build txns   │
        │  parse_directory()       batch + incremental driver   │
        │  split_other_line()      tokenize non-transaction text│
        │  pieces_to_columns()     name each token              │
        │  build_statement_info()  merge fields into one row    │
        └───────┬───────────────────────────────┬───────────────┘
                │ uses                          │ produces
                ▼                               ▼
   ┌────────────────────────┐      ┌─────────────────────────────┐
   │  statement_parser      │      │   statement_parser.excel    │
   │  patterns.py (regex)   │      │   read_workbook()           │
   │  utils.py   (helpers)  │      │   merge_frames/merge_errors │
   └────────────────────────┘      │   write_workbook() + style  │
                                   └─────────────────────────────┘
```

| Module | Responsibility |
|---|---|
| `patterns.py` | All regex constants. Change statement layouts here. |
| `utils.py` | `to_number` (`"1,234.50"` → `1234.5`, trailing `-` = negative), `clean` (collapse whitespace), `unique_name` (`x` → `x_2`, `x_3`, …). |
| `parsing.py` | Everything that touches statement text: header, line classification, other-field tokenizing/naming, batch driver. |
| `excel.py` | Workbook I/O only: read previous output, merge frames, write + style sheets. |
| `parse_to_json_v3.py` | Orchestration: CLI flags, skip/continue decision, summaries, exit codes. |

---

## Execution flow

### 1. Single file

```
main()
 ├─ resolve_paths()                src + default out (<stem>_v3.xlsx)
 ├─ load_existing_workbook(out)    None when --fresh or file absent
 └─ run_single()
     ├─ already processed? (src.name in Source_File registry) → stop
     ├─ parse_statement(src)
     │    ├─ read_header(lines)            owner name / address / account no
     │    └─ classify every line (see §3)  → transactions + other_lines
     ├─ build_statement_info(header, other_lines, header_lines,
     │                       counter, used, file_used)
     │    └─ per line: split_other_line() → pieces_to_columns()
     └─ write_workbook(out, df_txn, df_info, errors)
```

### 2. Directory (batch + incremental)

```
main()
 └─ run_directory()
     ├─ fresh?  parse_directory(src, pattern, exclude=out)
     └─ else    parse_directory(src, pattern,
                    skip_files      = processed_files(workbook),   # registry
                    counter_start   = placeholder_start(workbook), # max header_N + 1
                    known_columns   = existing_columns(workbook),  # label reuse
                    exclude         = out)
     ├─ per file: parse_statement() → tag rows with Source_File
     ├─ errors collected per file; the run never aborts on one bad file
     ├─ merge:  merge_frames(old, new) per sheet, merge_errors(...)
     └─ write_workbook(...)  # rewrites the whole file (also reapplies styling)
```

Incremental rules:

- A file counts as **processed** when its relative path appears in the
  `Source_File` column of `Transactions` or `Statement_Info`.
- Files recorded in `Errors` are **not** registered → they are retried next run.
- Placeholder numbering **continues** after the highest existing `header_N`
  (reusing `header_1` for a new file would pour an unrelated value into an
  existing column).
- Labels found in the existing workbook are **reused** as the same column.
- Nothing new → `Nothing to do`, the workbook file is not touched.
- An existing workbook without `Source_File` tracking aborts with guidance
  to use `--fresh`.

### 3. Line classification inside `parse_statement()`

```
line ──┬─ starts with two dates (TXN_START) ──────────► transaction
       │      amount/flag pulled from line end (AMOUNT_AT_END);
       │      no amount → pending, completed by a later line
       ├─ "XX. ..." at start (SECTION_RE) ────────────► section header
       │      (also recorded in Statement_Info)
       ├─ only a card number (CARD_ONLY) ─────────────► new card context
       │      (also recorded in Statement_Info)
       ├─ contains a card number ─► "CARD … SUB TOTAL -TYPE amount"
       │                            (SUB_TOTAL_PATTERN) → transaction
       │                            any other card line → other data
       └─ anything else ──────────────────────────────► other data
```

`Section` and `Card_Number` are *context*: every transaction inherits the last
section header and last card-only line seen before it.

### 4. Other fields: tokens → columns

```
line ─► split_other_line() ─► ordered pieces
        OTHER_TOKEN = (CARD) | (DATE) | (NUMBER_TOKEN), left-to-right;
        the text between matches becomes ("text", ...) pieces

      ─► pieces_to_columns()
           exactly one text piece BEFORE all values?
             yes → that text is the label      → column "TOTAL - DEBITS"
             no  → every piece gets header_N   → global counter, never reused

      ─► build_statement_info() merges header + all pieces into one dict/row
```

Examples:

| Line | Pieces | Columns |
|---|---|---|
| `TOTAL - DEBITS   4,940.48` | text, number | `TOTAL - DEBITS = 4940.48` |
| `080,040.67  8,181.13` | number, number | `header_7 = 80040.67`, `header_8 = 8181.13` |
| `MR. JITRJL SUQBGHVF …` | text only | `header_12 = "MR. JITRJL …"` |
| `5364 2160 0214 2777` | card | `header_14 = "5364 2160 0214 2777"` |

## Column naming rules (`Statement_Info`)

| Situation | Result |
|---|---|
| One text piece before the values | the text is the column name (e.g. `TOTAL - DEBITS`) |
| No usable label | `header_1`, `header_2`, … — **globally unique across all files and runs** |
| Same label in another file / previous run | **same column is reused** (shared semantics) |
| Same label twice within one file | `LABEL`, `LABEL_2` |
| Label equals a reserved name (`Owner_Name`, `Source_File`, …) or looks like `header_N` | suffixed `_2`, never overwrites |
| A file lacks a field another file has | cell left empty |

## Regex reference (`patterns.py`)

| Name | Pattern | Purpose |
|---|---|---|
| `DATE` | `\d{2}/\d{2}/\d{2,4}` | dd/mm/yy(yy); not validated |
| `AMOUNT` | `-?\d{1,3}(?:,\d{3})*(?:\.\d{2})-?` | value **with cents**; leading or trailing minus |
| `CARD` | `\d{4}\s\d{4}\s\d{4}\s\d{4}` | 16 digits in four space-separated groups |
| `NUMBER_TOKEN` | `-?\d{1,3}(?:,\d{3})*(?:\.\d{1,2})-?` | looser amount for other-field tokenizing (cents optional → catches `707,841`) |
| `TXN_START` | `^\s*(DATE)\s+(DATE)\s*(.*)$` | transaction + posting date at line start |
| `AMOUNT_AT_END` | `(?:^\|\s)(AMOUNT)\s*([A-Za-z]{0,3})\s*$` | amount + optional flag (`CR`, `NS`) at line end |
| `SUB_TOTAL_PATTERN` | card + `SUB TOTAL` + `-TYPE` + amount | card sub-total rows |
| `SECTION_RE` | `^\s*[A-Z]{2}\.\s+\S` | section headers like `PU. PURCHASES` |
| `OTHER_TOKEN` | `(CARD)\|(DATE)\|(NUMBER)` | ordered alternation for tokenizing |

## Making changes

- **Layout differs** → adjust the patterns in `patterns.py`; nothing else should need to change.
- **Know what a `header_N` means** → rename it once values are identified, e.g. in
  `parsing.py` after `build_statement_info()`:

  ```python
  RENAME = {"header_7": "Previous_Balance", "header_8": "Payments", ...}
  info = {RENAME.get(k, k): v for k, v in info.items()}
  ```

  Labeled columns rename themselves from the source text.
- **New transaction column** → add the key to the `rec` dict in `parse_statement()`
  and (if computed later) fill it where the value becomes known.
- **Different file type** → pass `--pattern` (e.g. `--pattern "*.csv"`); discovery,
  skipping and merging are pattern-agnostic.

## Known limitations

- One statement per `.txt` file (the parser has no multi-statement awareness).
- File identity for incremental runs is the **relative path** — re-running with a
  *different* parent directory re-parses the same files. A content hash would fix this.
- If a processed file is **edited later**, it is skipped (its path is registered);
  use `--fresh` to reparse current content.
- `--fresh` re-numbers placeholders by the current file sort order, so `header_N`
  numbers can shift between a fresh run and an incremental history (values never mix).
- The owner-address filter uses the strict `AMOUNT` (cents required): a card-only
  line without cents can end up classified as part of the address.
- Dates are pattern-matched, not validated — the dummy data intentionally
  contains impossible dates.
