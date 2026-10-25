"""Regex patterns and shared constants used to parse the statement text."""

import re

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

# AMOUNT requires the ".dd" cents part, so it misses values such as
# "707,841" or "564,023". This looser pattern is used ONLY when splitting
# non-transaction lines; the transaction logic keeps using AMOUNT as-is.
NUMBER_TOKEN = r"-?\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?-?"
OTHER_TOKEN = re.compile(rf"({CARD})|({DATE})|({NUMBER_TOKEN})")

# Prefix for fields that have no meaningful label in the source line.
OTHER_FIELD_PREFIX = "header_"
