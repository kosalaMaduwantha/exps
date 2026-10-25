"""Statement header / owner block detection."""

from .patterns import ALL_AMOUNTS, ALL_DATES, CARD_RE
from .utils import clean


def read_header(lines):
    """Title = first non-empty line; owner block = next consecutive non-empty
    lines (first line = owner name, the rest = address).

    Returns (header_dict, header_line_numbers). header_line_numbers are the
    1-based lines whose content is already represented in header_dict, so
    they must not be repeated in the other-data columns.
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

    # keep only the block before the first line containing amounts/dates
    kept = [
        (n, b) for n, b in block
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
