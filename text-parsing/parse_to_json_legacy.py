"""
Parse a (text-based) bank credit statement into an Excel workbook.

Usage:
    python parse_statement_to_excel.py replaced_dummy_document.txt statement.xlsx

Requires: pandas, openpyxl   (pip install pandas openpyxl)

Output sheets
    Transactions   - one row per transaction, each tagged with the statement owner
    Statement_Info - owner / account details found in the header
    Other_Lines    - every non-transaction line that contains amounts/dates
                     (balances, subtotals, summary rows) so nothing is lost
"""

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
    lines (first line = owner name, the rest = address)."""
    idx = [i for i, l in enumerate(lines) if l.strip()]
    if not idx:
        return {}
    name = clean(lines[idx[0]])

    block, prev = [], None
    for i in idx[1:]:
        if prev is not None and i != prev + 1:
            break  # blank line -> end of owner block
        block.append(clean(lines[i]))
        prev = i
    print("printing the block: ", block)
    # keep only the block before the first line containing amounts/dates
    block = [b for b in block if not (ALL_AMOUNTS.search(b) or ALL_DATES.search(b))]

    owner_address = " | ".join(block[0:])
    # Statement account number = first card-like number on a line that also has a date
    account = ""
    for l in lines:
        if ALL_DATES.search(l):
            m = CARD_RE.search(l)
            if m:
                account = m.group(1)
                break
    return {
        "Owner_Name": name,
        "Owner_Address": owner_address,
        "Statement_Account_No": account,
    }


# --------------------------------------------------------------------------
# Main parsing
# --------------------------------------------------------------------------
def parse_statement(path: Path):
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header = read_header(lines)
    owner_key = header.get("Statement_Account_No")

    transactions, others = [], []
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
        if SECTION_RE.match(line) and not ALL_AMOUNTS.search(line.split()[-1] if line.split() else ""):
            current_section = clean(re.sub(AMOUNT, "", line))
            others.append(_other_row(lineno, line, header, owner_key, current_card))
            continue

        # ---- line containing only a card number -> new card context --------
        m = CARD_ONLY.match(line)
        if m:
            finish(pending)
            pending = None
            current_card = m.group(1)
            continue

        # ---- continuation line (amount / reference for a transaction) ------
        # target = pending if pending is not None else (transactions[-1] if transactions else None)
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
                # card number + text after a transaction => reference info
                m = re.match(SUB_TOTAL_PATTERN, text)
                if not m:
                    continue
                new_transaction["Transaction_Type"] = "DEBIT" if m.group(2).strip() else ""
                new_transaction["Description"] = clean(m.group(1).strip())
                new_transaction["Amount"] = to_number(m.group(3))
            else:
                new_transaction["Description"] = (new_transaction["Description"] + " " + text).strip()
            if pending is not None and new_transaction["Amount"] is not None:
                finish(pending)
                pending = None
            transactions.append(new_transaction)
            continue

        # ---- anything else: summary rows, balances, subtotals, noise -------
        if cm:
            current_card = cm.group(1) if CARD_ONLY.match(line) else current_card
        others.append(_other_row(lineno, line, header, owner_key, current_card))

    finish(pending)
    return header, transactions, others


def _other_row(lineno, line, header, owner_key, card):
    return {
        "Owner_Key": owner_key,
        "Card_Number": card,
        "Source_Line": lineno,
        "Text": clean(line),
        "Dates_Found": ", ".join(ALL_DATES.findall(line)),
        "Amounts_Found": ", ".join(ALL_AMOUNTS.findall(line)),
    }


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
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".xlsx")

    header, txns, others = parse_statement(src)

    df_txn = pd.DataFrame(txns)
    df_info = pd.DataFrame(list(header.items()), columns=["Field", "Value"])
    df_other = pd.DataFrame(others)

    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        df_txn.to_excel(xw, sheet_name="Transactions", index=False)
        df_info.to_excel(xw, sheet_name="Statement_Info", index=False)
        df_other.to_excel(xw, sheet_name="Other_Lines", index=False)
        style_sheet(xw.sheets["Transactions"], money_cols=("Amount",))
        style_sheet(xw.sheets["Statement_Info"])
        style_sheet(xw.sheets["Other_Lines"])

    print(f"Parsed {len(df_txn)} transactions -> {out}")
    if not df_txn.empty and df_txn["Amount"].isna().any():
        print("Warning: some transactions have no amount - check 'Source_Line' in the sheet.")


if __name__ == "__main__":
    main()