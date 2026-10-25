"""
Parse a (text-based) bank credit statement into an Excel workbook.

Usage:
    python parse_to_json_v2.py sample_statement.txt sample_statement_output_v2.xlsx

Requires: pandas, openpyxl   (pip install pandas openpyxl)

Output sheets
    Transactions   - one row per transaction, each tagged with the statement owner
    Statement_Info - ONE row, wide layout. Owner/account details plus every other
                     piece of data found in the statement. Columns that have a
                     meaningful label in the source line use that label
                     (e.g. "TOTAL - DEBITS"); unlabeled values fall back to
                     header_1, header_2, ...

This is a copy of parse_to_json.py with the "other data" handling changed:
    * other data is merged into Statement_Info instead of a separate sheet
    * Statement_Info is a single row with one column per field
    * lines that were previously dropped (card line that is not SUB TOTAL)
      are now captured
    * card-only lines are captured as data as well
"""

import itertools
import re
import sys
from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter

# --------------------------------------------------------------------------
# Regex patterns - tweak these if your real statement layout differs
# --------------------------------------------------------------------------
DATE = r"\d{2}/\d{2}/\d{2,4}"
AMOUNT = r"-?\d{1,3}(?:,\d{3})*(?:\.\d{2})-?"
CARD = r"\d{4}\s\d{4}\s\d{4}\s\d{4}"

# Transaction line: two dates at the start (transaction date + posting date)
TXN_START = re.compile(rf"^\s*({DATE})\s+({DATE})\s*(.*)$")
# Amount at the END of a line, optionally followed by a short flag (e.g. 'CR', 'NS')
AMOUNT_AT_END = re.compile(rf"(?:^|\s)({AMOUNT})\s*([A-Za-z]{{0,3}})\s*$")
CARD_RE = re.compile(rf"\b({CARD})\b")
CARD_ONLY = re.compile(rf"^\s*({CARD})\s*$")
# Section headers such as "XU. SOMETHING ..." (two letters + dot at line start)
SECTION_RE = re.compile(r"^\s*[A-Z]{2}\.\s+\S")
ALL_AMOUNTS = re.compile(AMOUNT)
ALL_DATES = re.compile(DATE)

SUB_TOTAL_PATTERN = rf"^\s*({CARD}\s+SUB\s+TOTAL)\s+-([A-Z]+)\s+({AMOUNT})\s*$"

# --------------------------------------------------------------------------
# "Other data" tokenizer
# --------------------------------------------------------------------------
# AMOUNT requires the ".dd" cents part, so it misses values such as
# "707,841" or "564,023". This looser pattern is used ONLY when splitting
# non-transaction lines; the transaction logic keeps using AMOUNT as-is.
NUMBER_TOKEN = r"-?\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?-?"
OTHER_TOKEN = re.compile(rf"({CARD})|({DATE})|({NUMBER_TOKEN})")

# Prefix for fields that have no meaningful label in the source line.
# Rename the resulting column names here (or map them later) once the real
# meaning of each value is known.
OTHER_FIELD_PREFIX = "header_"


def to_number(text: str) -> float:
    """'1,234.50' -> 1234.5 ; trailing '-' means negative."""
    neg = text.startswith("-") or text.endswith("-")
    value = float(text.strip("-").replace(",", ""))
    return -value if neg else value


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------
# Header / owner detection
# --------------------------------------------------------------------------
def read_header(lines):
    """Title = first non-empty line; owner block = next consecutive non-empty
    lines (first line = owner name, the rest = address).

    Returns (header_dict, header_line_numbers) where header_line_numbers are
    the 1-based line numbers whose content is already represented in
    header_dict (so they must not be duplicated in the other-data columns).
    """
    idx = [i for i, l in enumerate(lines) if l.strip()]
    if not idx:
        return {}, set()
    name_line = idx[0] + 1
    name = clean(lines[idx[0]])

    block, prev = [], None
    for i in idx[1:]:
        if prev is not None and i != prev + 1:
            break  # blank line -> end of owner block
        block.append((i + 1, clean(lines[i])))
        prev = i
    print("printing the block: ", [b for _, b in block])
    # keep only the block before the first line containing amounts/dates
    kept = [
        (n, b)
        for n, b in block
        if not (ALL_AMOUNTS.search(b) or ALL_DATES.search(b))
    ]

    owner_address = " | ".join(b for _, b in kept)
    # Statement account number = first card-like number on a line that also has a date
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


# --------------------------------------------------------------------------
# Splitting "other" lines into independent pieces
# --------------------------------------------------------------------------
def split_other_line(line):
    """Split a non-transaction line into ordered pieces.

    Returns a list of (kind, value) tuples where kind is one of
    'card', 'date', 'number' or 'text'. Recognized values are taken out of
    the line; the text between/around them is kept as 'text' pieces.
    """
    pieces, pos = [], 0
    for m in OTHER_TOKEN.finditer(line):
        gap = line[pos:m.start()]
        if gap.strip():
            pieces.append(("text", clean(gap)))
        if m.group(1) is not None:
            pieces.append(("card", clean(m.group(1))))
        elif m.group(2) is not None:
            pieces.append(("date", m.group(2)))
        else:
            pieces.append(("number", m.group(0)))
        pos = m.end()
    tail = line[pos:]
    if tail.strip():
        pieces.append(("text", clean(tail)))
    return pieces


def _convert(kind, value):
    if kind == "number":
        return to_number(value)
    return value


def _unique(name, used):
    if name not in used:
        return name
    i = 2
    while f"{name}_{i}" in used:
        i += 1
    return f"{name}_{i}"


def pieces_to_columns(pieces, counter, used):
    """Turn one line's pieces into (column_name, value) pairs.

    Rules:
      * exactly one text piece BEFORE all values -> use that text as the
        column name for the first value (e.g. "TOTAL - DEBITS")
      * anything unlabeled -> header_1, header_2, ... (global counter)
      * duplicate names -> suffix _2, _3, ...
    """
    values = [p for p in pieces if p[0] != "text"]
    texts = [(i, p) for i, p in enumerate(pieces) if p[0] == "text"]

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
        for i, (kind, value) in enumerate(values):
            name = label if i == 0 else f"{label}_{i + 1}"
            name = _unique(name, used)
            used.add(name)
            out.append((name, _convert(kind, value)))
    else:
        for kind, value in pieces:
            name = _unique(f"{OTHER_FIELD_PREFIX}{next(counter)}", used)
            used.add(name)
            out.append((name, _convert(kind, value)))
    return out


def build_statement_info(header, other_lines, header_lines):
    """Merge the owner/account header and every other line into ONE dict."""
    info = dict(header)  # preserves Owner_Name, Owner_Address, Account order
    used = set(info)
    counter = itertools.count(1)
    for lineno, line in other_lines:
        if lineno in header_lines:
            continue  # already represented in Owner_Name / Owner_Address
        for name, value in pieces_to_columns(
            split_other_line(line), counter, used
        ):
            info[name] = value
    return info


# --------------------------------------------------------------------------
# Main parsing
# --------------------------------------------------------------------------
def parse_statement(path: Path):
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header, header_lines = read_header(lines)
    owner_key = header.get("Statement_Account_No")

    transactions, other_lines = [], []
    current_card = header.get("Statement_Account_No", "")
    current_section = ""
    pending = None  # transaction waiting for its amount (amount on a later line)

    def finish(rec):
        if rec is not None:
            rec["Description"] = clean(rec["Description"])
            transactions.append(rec)

    for lineno, raw in enumerate(lines, start=1):
        if not raw.strip():
            continue
        line = raw.rstrip()

        # ---- new transaction line ------------------------------------------
        m = TXN_START.match(line)
        if m:
            finish(pending)
            pending = None
            txn_date, post_date, rest = m.groups()
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
                print("is there any record with no amount there")
                pending = rec  # amount probably on a following line
            else:
                transactions.append({**rec, "Description": clean(rec["Description"])})
            continue

        # ---- section header ------------------------------------------------
        if SECTION_RE.match(line) and not ALL_AMOUNTS.search(
            line.split()[-1] if line.split() else ""
        ):
            current_section = clean(re.sub(AMOUNT, "", line))
            other_lines.append((lineno, line))
            continue

        # ---- line containing only a card number -> new card context --------
        m = CARD_ONLY.match(line)
        if m:
            finish(pending)
            pending = None
            other_lines.append((lineno, line))  # capture it as data too
            current_card = m.group(1)
            continue

        # ---- card line / continuation line ---------------------------------
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
        if new_transaction is not None and (pending is not None or cm):
            amt = AMOUNT_AT_END.search(line)
            text = line
            if amt and new_transaction["Amount"] is None:
                new_transaction["Amount"] = to_number(amt.group(1))
                new_transaction["Amount_Flag"] = amt.group(2)
                text = line[: amt.start()]
            if cm:
                # card number + text after a transaction => SUB TOTAL row
                m = re.match(SUB_TOTAL_PATTERN, text)
                if not m:
                    # Not a SUB TOTAL row: keep the line instead of dropping it
                    other_lines.append((lineno, line))
                    continue
                new_transaction["Transaction_Type"] = "DEBIT" if m.group(2).strip() else ""
                new_transaction["Description"] = clean(m.group(1).strip())
                new_transaction["Amount"] = to_number(m.group(3))
            else:
                new_transaction["Description"] = (
                    new_transaction["Description"] + " " + text
                ).strip()
            if pending is not None and new_transaction["Amount"] is not None:
                finish(pending)
                pending = None
            transactions.append(new_transaction)
            continue

        # ---- anything else: summary rows, balances, subtotals, noise -------
        if cm:
            current_card = cm.group(1) if CARD_ONLY.match(line) else current_card
        other_lines.append((lineno, line))

    finish(pending)
    return header, transactions, other_lines, header_lines


# --------------------------------------------------------------------------
# Excel output
# --------------------------------------------------------------------------
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
        width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
        ws.column_dimensions[get_column_letter(col_idx)].width = min(max(width + 2, 10), 60)
    headers = [c.value for c in ws[1]]
    for name in money_cols:
        if name in headers:
            col = headers.index(name) + 1
            for r in range(2, ws.max_row + 1):
                ws.cell(row=r, column=col).number_format = "#,##0.00;(#,##0.00);-"
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    src = Path(sys.argv[1])
    out = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else src.with_name(f"{src.stem}_v2.xlsx")
    )

    header, txns, other_lines, header_lines = parse_statement(src)
    info = build_statement_info(header, other_lines, header_lines)

    df_txn = pd.DataFrame(txns)
    df_info = pd.DataFrame([info], columns=list(info.keys()))

    money_cols = [
        k for k, v in info.items() if isinstance(v, (int, float))
    ]

    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        df_txn.to_excel(xw, sheet_name="Transactions", index=False)
        df_info.to_excel(xw, sheet_name="Statement_Info", index=False)
        style_sheet(xw.sheets["Transactions"], money_cols=("Amount",))
        style_sheet(xw.sheets["Statement_Info"], money_cols=tuple(money_cols))

    print(f"Parsed {len(df_txn)} transactions and {len(info)} statement fields -> {out}")
    if not df_txn.empty and df_txn["Amount"].isna().any():
        print("Warning: some transactions have no amount - check 'Source_Line' in the sheet.")


if __name__ == "__main__":
    main()
