"""All statement parsing logic in one place.

This module is the single home for parsing the statement text:

    read_header()          - read the owner/account header block
    split_other_line()     - tokenize a non-transaction line into pieces
    _convert()             - turn a numeric piece into a float, keep others text
    pieces_to_columns()    - give each piece a column name (label or header_N)
    build_statement_info() - merge header fields + other fields into one row
    parse_statement()      - classify every line and build transaction records

Regex constants live in patterns.py and small helpers in utils.py.
"""

import itertools
import re
from pathlib import Path

from .patterns import (
    ALL_AMOUNTS,
    ALL_DATES,
    AMOUNT,
    AMOUNT_AT_END,
    CARD_ONLY,
    CARD_RE,
    OTHER_FIELD_PREFIX,
    OTHER_TOKEN,
    SECTION_RE,
    SUB_TOTAL_PATTERN,
    TXN_START,
)
from .utils import clean, to_number, unique_name

# Fields every statement-info row starts with, plus the batch bookkeeping
# column. Discovered labels must never silently take one of these names.
BASE_FIELD_NAMES = ("Owner_Name", "Owner_Address", "Statement_Account_No")
RESERVED_FIELD_NAMES = BASE_FIELD_NAMES + ("Source_File",)

# Matches generated placeholder names like "header_17". A label that happens
# to look like a placeholder must not reuse that column.
_PLACEHOLDER_NAME = re.compile(rf"{re.escape(OTHER_FIELD_PREFIX)}\d+")


# ==========================================================================
# Header / owner block
# ==========================================================================
def read_header(lines):
    """Read the statement title and the owner/account block.

    How it works:
      * The first non-empty line is the owner name.
      * The next consecutive non-empty lines are the address block. A blank
        line ends the block.
      * Address lines that contain a date or an amount are dropped, because
        they belong to statement data rather than the postal address.
      * The account number is the first card-like number found on a line that
        also contains a date.

    Returns:
        (header, header_lines)
        header: dict with Owner_Name, Owner_Address, Statement_Account_No.
        header_lines: 1-based line numbers whose content is already covered by
            header. The other-field builder skips these to avoid duplicates.
    """
    # Positions of every non-empty line; the first one is the owner name.
    idx = [i for i, l in enumerate(lines) if l.strip()]
    if not idx:
        return {}, set()
    name_line = idx[0] + 1
    name = clean(lines[idx[0]])

    # Collect consecutive non-empty lines after the name. Once a blank line
    # appears (i != prev + 1), the owner block is finished.
    block, prev = [], None
    for i in idx[1:]:
        if prev is not None and i != prev + 1:
            break
        block.append((i + 1, clean(lines[i])))
        prev = i

    # Keep only address-like lines (no amounts, no dates).
    kept = [
        (n, b)
        for n, b in block
        if not (ALL_AMOUNTS.search(b) or ALL_DATES.search(b))
    ]
    owner_address = " | ".join(b for _, b in kept)

    # Statement account number: first card-like number on a line with a date.
    account = ""
    for l in lines:
        if ALL_DATES.search(l):
            m = CARD_RE.search(l)
            if m:
                account = m.group(1)
                break

    header = {
        "Owner_Name": name,
        "Owner_Address": owner_address,
        "Statement_Account_No": account,
    }
    header_lines = {name_line} | {n for n, _ in kept}
    return header, header_lines


# ==========================================================================
# Other fields (non-transaction lines)
# ==========================================================================
def split_other_line(line):
    """Tokenize one non-transaction line into ordered (kind, value) pieces.

    The combined OTHER_TOKEN regex recognizes, in this priority order:
        1. CARD   - 4 groups of 4 digits separated by whitespace
        2. DATE   - dd/mm/yy or dd/mm/yyyy
        3. NUMBER - a looser amount pattern that also accepts values without
                    cents (e.g. "707,841"), because AMOUNT requires ".dd"

    Anything between those matches is kept as a ("text", ...) piece, which is
    what allows labels such as "TOTAL - DEBITS" to be recovered.

    Example:
        "TOTAL - DEBITS   4,940.48"
        -> [("text", "TOTAL - DEBITS"), ("number", "4,940.48")]
    """
    pieces, pos = [], 0
    for m in OTHER_TOKEN.finditer(line):
        # Text between the previous match and this one.
        gap = line[pos:m.start()]
        if gap.strip():
            pieces.append(("text", clean(gap)))

        # group(1)=CARD, group(2)=DATE, group(3)=NUMBER
        if m.group(1) is not None:
            pieces.append(("card", clean(m.group(1))))
        elif m.group(2) is not None:
            pieces.append(("date", m.group(2)))
        else:
            pieces.append(("number", m.group(0)))
        pos = m.end()

    # Any trailing text after the last match.
    tail = line[pos:]
    if tail.strip():
        pieces.append(("text", clean(tail)))
    return pieces


def _convert(kind, value):
    """Convert a tokenizer piece to its final value type.

    Numbers become floats; cards, dates and text stay as strings.
    """
    return to_number(value) if kind == "number" else value


def pieces_to_columns(pieces, counter, used, file_used=None):
    """Turn one line's pieces into (column_name, value) pairs.

    Naming rules:
      * If the line has exactly one text piece and it appears BEFORE all
        values, that text becomes the column name. This is how
        "TOTAL - DEBITS   4,940.48" becomes column "TOTAL - DEBITS".
      * Otherwise every piece gets a placeholder name from the shared
        counter: header_1, header_2, ...
      * If a name is already in `used`, unique_name() appends _2, _3, ...

    Two name spaces keep multi-file / incremental merging safe:
      * `used`     - every column name ever assigned (this run plus the
                     columns of an existing workbook on re-runs)
      * `file_used`- names assigned for the CURRENT file only
    A name already in `used` but NOT in `file_used` is REUSED instead of
    suffixed: that is how two files (or two runs) that both contain
    "TOTAL - DEBITS" share one column. Reserved names (base fields,
    "Source_File") and generated placeholder names are never reused.

    `counter` is shared across all lines/files so numbering continues instead
    of restarting for every line.
    """
    if file_used is None:
        file_used = used  # single-file mode: one namespace (old behavior)

    def assign(base):
        """Pick a column name for `base` and record it in both namespaces."""
        reusable = (
            base in used
            and base not in file_used
            and base not in RESERVED_FIELD_NAMES
            and not _PLACEHOLDER_NAME.fullmatch(base)
        )
        name = base if reusable else unique_name(base, used | file_used)
        used.add(name)
        file_used.add(name)
        return name

    values = [p for p in pieces if p[0] != "text"]
    texts = [(i, p) for i, p in enumerate(pieces) if p[0] == "text"]

    # A text piece is a usable label only when it comes before the values.
    # This prevents a trailing flag such as "CR" from becoming a column name.
    label = None
    if len(texts) == 1 and values:
        text_index, (_, text_value) = texts[0]
        first_value_index = next(
            i for i, p in enumerate(pieces) if p[0] != "text"
        )
        if text_index < first_value_index:
            label = text_value

    out = []
    if label:
        # Labeled line: first value uses the label, extras get label_2, ...
        for i, (kind, value) in enumerate(values):
            base = label if i == 0 else f"{label}_{i + 1}"
            out.append((assign(base), _convert(kind, value)))
    else:
        # Unlabeled line: header_1, header_2, ... in source order.
        for kind, value in pieces:
            out.append(
                (assign(f"{OTHER_FIELD_PREFIX}{next(counter)}"),
                 _convert(kind, value))
            )
    return out


def build_statement_info(header, other_lines, header_lines,
                         counter=None, used=None, file_used=None):
    """Merge the header fields and every other line into ONE flat dict.

    The returned dict is later written as a single row, so each key becomes a
    column in the Statement_Info sheet.

    Lines whose numbers are in `header_lines` are skipped because their
    content is already represented by Owner_Name / Owner_Address.

    The optional parameters let several files (and incremental re-runs) share
    one naming space:
      counter   - itertools.count shared across files; keeps header_N numbers
                  unique and continuing after an existing workbook
      used      - every column name ever assigned (mutated in place)
      file_used - per-file namespace; pass set(header) for each file
    With the defaults, an independent naming space is created (old behavior).
    """
    info = dict(header)  # keeps Owner_Name, Owner_Address, Account first
    if counter is None:
        counter = itertools.count(1)  # shared header_N counter
    if used is None:
        used = set(info)  # names that must not be reused
    used.update(header)   # base names are reserved in every namespace

    for lineno, line in other_lines:
        if lineno in header_lines:
            continue
        for name, value in pieces_to_columns(
            split_other_line(line), counter, used, file_used
        ):
            info[name] = value
    return info


# ==========================================================================
# Line classification and transaction parsing
# ==========================================================================
def parse_statement(path: Path):
    """Parse a statement file into header, transactions and other lines.

    Each non-empty line is classified in this order:
      1. TXN_START       - starts with two dates -> a transaction
      2. SECTION_RE      - "XX. ..." section header
      3. CARD_ONLY       - the line is only a card number -> new card context
      4. card/continuation line - SUB TOTAL rows become transactions; other
         card-bearing lines are kept as other data
      5. everything else - summary rows, balances, etc.

    Returns:
        (header, transactions, other_lines, header_lines)
    """
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header, header_lines = read_header(lines)
    owner_key = header.get("Statement_Account_No")

    transactions, other_lines = [], []
    current_card = header.get("Statement_Account_No", "")
    current_section = ""
    pending = None  # transaction waiting for its amount (amount on a later line)

    def finish(rec):
        """Normalize and store a pending transaction record."""
        if rec is not None:
            rec["Description"] = clean(rec["Description"])
            transactions.append(rec)

    for lineno, raw in enumerate(lines, start=1):
        if not raw.strip():
            continue
        line = raw.rstrip()

        # ---- 1. new transaction line --------------------------------------
        # Two dates at the start: transaction date and posting date.
        m = TXN_START.match(line)
        if m:
            finish(pending)
            pending = None
            txn_date, post_date, rest = m.groups()

            # Try to pull the amount (and an optional flag such as CR/NS)
            # from the end of the line.
            amt = AMOUNT_AT_END.search(rest)
            amount, flag = (None, "")
            if amt:
                amount, flag = to_number(amt.group(1)), amt.group(2)
                rest = rest[: amt.start()]

            rec = {
                "Statement_Account_No": header.get("Statement_Account_No", ""),
                "Owner_Key": owner_key,
                "Section": current_section,
                "Card_Number": current_card,
                "Txn_Date": txn_date,
                "Post_Date": post_date,
                "Description": rest,
                "Transaction_Type": "",
                "Amount": amount,
                "Amount_Flag": flag,
                "Source_Line": lineno,
            }
            if amount is None:
                # Amount probably on a following line; finish it later.
                pending = rec
            else:
                transactions.append(
                    {**rec, "Description": clean(rec["Description"])}
                )
            continue

        # ---- 2. section header --------------------------------------------
        # "XX. TEXT" starts a new section. Ignore it if the last token looks
        # like an amount, which means this is really a data row.
        if SECTION_RE.match(line) and not ALL_AMOUNTS.search(
            line.split()[-1] if line.split() else ""
        ):
            current_section = clean(re.sub(AMOUNT, "", line))
            other_lines.append((lineno, line))
            continue

        # ---- 3. card-only line -> new card context ------------------------
        m = CARD_ONLY.match(line)
        if m:
            finish(pending)
            pending = None
            other_lines.append((lineno, line))  # capture it as data too
            current_card = m.group(1)
            continue

        # ---- 4. card line / continuation line -----------------------------
        # Build the record shape up front; it is used only if the line turns
        # out to be a SUB TOTAL transaction.
        new_transaction = {
            "Statement_Account_No": header.get("Statement_Account_No", ""),
            "Owner_Key": owner_key,
            "Section": current_section,
            "Card_Number": current_card,
            "Txn_Date": "",
            "Post_Date": "",
            "Description": "",
            "Transaction_Type": "",
            "Amount": "",
            "Amount_Flag": "",
            "Source_Line": lineno,
        }
        cm = CARD_RE.search(line)
        if pending is not None or cm:
            amt = AMOUNT_AT_END.search(line)
            text = line
            if amt and new_transaction["Amount"] is None:
                new_transaction["Amount"] = to_number(amt.group(1))
                new_transaction["Amount_Flag"] = amt.group(2)
                text = line[: amt.start()]
            if cm:
                # Card + "SUB TOTAL -TYPE amount" is a transaction row.
                m = re.match(SUB_TOTAL_PATTERN, text)
                if not m:
                    # Any other card-bearing line is statement data, not a
                    # transaction, so keep it instead of dropping it.
                    other_lines.append((lineno, line))
                    continue
                new_transaction["Transaction_Type"] = (
                    "DEBIT" if m.group(2).strip() else ""
                )
                new_transaction["Description"] = clean(m.group(1).strip())
                new_transaction["Amount"] = to_number(m.group(3))
            else:
                # Continuation text for a pending transaction.
                new_transaction["Description"] = (
                    new_transaction["Description"] + " " + text
                ).strip()
            if pending is not None and new_transaction["Amount"] is not None:
                finish(pending)
                pending = None
            transactions.append(new_transaction)
            continue

        # ---- 5. anything else ---------------------------------------------
        # Summary rows, balances, totals, noise.
        if cm:
            current_card = cm.group(1) if CARD_ONLY.match(line) else current_card
        other_lines.append((lineno, line))

    finish(pending)
    return header, transactions, other_lines, header_lines


# ==========================================================================
# Batch mode: recursively parse a directory of statement files
# ==========================================================================
def discover_statement_files(parent, pattern="*.txt", exclude=None):
    """Recursively find candidate statement files under `parent`.

    Guarantees a deterministic result:
      * sorted by relative POSIX path (stable header_N numbering across runs)
      * duplicates removed via the resolved absolute path (symlink safety)
      * hidden entries (".git", ".hidden.txt", ...) are skipped
      * `exclude` (e.g. the output workbook itself) is skipped
    """
    parent = Path(parent)
    exclude_resolved = Path(exclude).resolve() if exclude else None
    seen, found = set(), []
    for path in parent.rglob(pattern):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(parent).parts
        if any(part.startswith(".") for part in rel_parts):
            continue  # hidden directories/files (e.g. .git)
        resolved = path.resolve()
        if resolved in seen:
            continue
        if exclude_resolved is not None and resolved == exclude_resolved:
            continue
        seen.add(resolved)
        found.append(path)
    return sorted(found, key=lambda p: p.relative_to(parent).as_posix())


def parse_directory(parent, pattern="*.txt", skip_files=frozenset(),
                    counter_start=1, known_columns=(), exclude=None):
    """Parse every NEW statement file under `parent`, recursively.

    Used for batch and incremental runs:
      skip_files    - relative paths already recorded in an existing workbook
      counter_start - first free header_N index (max existing + 1)
      known_columns - column names of the existing workbook; labels that match
                      them are reused instead of suffixed
      exclude       - path to ignore (typically the output workbook itself)

    Returns a dict:
      transactions - row dicts, each starting with "Source_File"
      info_rows    - one statement-info dict per successfully parsed file
      errors       - [(Source_File, message), ...]
      parsed       - [(Source_File, txn_count, field_count), ...]
      found        - number of files discovered (before skipping)
    """
    parent = Path(parent)
    counter = itertools.count(counter_start)
    # Seed with known columns + base fields so label collisions are handled
    # identically in every file and every run.
    global_used = set(known_columns) | set(BASE_FIELD_NAMES)
    transactions, info_rows, errors, parsed = [], [], [], []
    found = 0

    for path in discover_statement_files(parent, pattern, exclude=exclude):
        found += 1
        rel = path.relative_to(parent).as_posix()
        if rel in skip_files:
            continue  # already recorded in an earlier run
        try:
            header, txns, others, header_lines = parse_statement(path)
            if not header and not txns and not others:
                errors.append((rel, "no data found"))
                continue
            info = build_statement_info(
                header, others, header_lines,
                counter=counter,
                used=global_used,
                file_used=set(header),
            )
            transactions.extend({"Source_File": rel, **t} for t in txns)
            info_rows.append({"Source_File": rel, **info})
            parsed.append((rel, len(txns), len(info)))
        except Exception as exc:  # report and keep going; never abort the batch
            errors.append((rel, f"{type(exc).__name__}: {exc}"))

    return {
        "transactions": transactions,
        "info_rows": info_rows,
        "errors": errors,
        "parsed": parsed,
        "found": found,
    }
