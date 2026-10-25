"""Statement line classification and transaction extraction."""

import re
from pathlib import Path

from .header import read_header
from .patterns import (
    ALL_AMOUNTS,
    AMOUNT,
    AMOUNT_AT_END,
    CARD_ONLY,
    CARD_RE,
    SECTION_RE,
    SUB_TOTAL_PATTERN,
    TXN_START,
)
from .utils import clean, to_number


def parse_statement(path: Path):
    """Parse the statement text.

    Returns (header, transactions, other_lines, header_lines) where
    other_lines is a list of (lineno, raw_line) for lines that are not
    transactions and still need to be turned into statement-info columns.
    """
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
        if pending is not None or cm:
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
