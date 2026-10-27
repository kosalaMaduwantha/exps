"""Function-based statement parsing package (no classes).

Excel workbook IO (statement_parser.excel) is retired from the package
surface: the extractor now writes streaming CSV outputs via csv_io.
"""

from .parsing import (
    build_statement_info,
    discover_statement_files,
    parse_directory,
    parse_statement,
    read_header,
    split_other_line,
)

__all__ = [
    "build_statement_info",
    "discover_statement_files",
    "parse_directory",
    "parse_statement",
    "read_header",
    "split_other_line",
]
