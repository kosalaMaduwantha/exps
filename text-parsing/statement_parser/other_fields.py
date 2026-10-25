"""Turn non-transaction lines into labeled or header_N columns."""

import itertools

from .patterns import OTHER_FIELD_PREFIX, OTHER_TOKEN
from .utils import clean, to_number, unique_name


def split_other_line(line):
    """Split a non-transaction line into ordered (kind, value) pieces.

    kind is 'card', 'date', 'number' or 'text'. Recognized values are
    removed from the line; text between/around them becomes 'text' pieces.
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
    return to_number(value) if kind == "number" else value


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
            name = unique_name(name, used)
            used.add(name)
            out.append((name, _convert(kind, value)))
    else:
        for kind, value in pieces:
            name = unique_name(f"{OTHER_FIELD_PREFIX}{next(counter)}", used)
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
