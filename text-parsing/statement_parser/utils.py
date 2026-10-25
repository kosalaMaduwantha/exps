"""Small pure helpers shared by the parser modules."""

import re


def to_number(text: str) -> float:
    """'1,234.50' -> 1234.5 ; trailing '-' means negative."""
    neg = text.startswith("-") or text.endswith("-")
    value = float(text.strip("-").replace(",", ""))
    return -value if neg else value


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def unique_name(name: str, used: set[str]) -> str:
    """Return name, or name_2 / name_3 / ... if it is already in used."""
    if name not in used:
        return name
    i = 2
    while f"{name}_{i}" in used:
        i += 1
    return f"{name}_{i}"
