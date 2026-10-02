"""
extraction/tidy.py -- deterministic clean-up of transcribed table cells.

Runs on the text a recogniser returns (no model calls):
  * ditto marks ("  〃  ^  do ...) are replaced by the value above them in
    the same column, as a reader of the ledger would understand them;
  * in columns that hold numbers, a letter O written for zero becomes 0.
"""

from __future__ import annotations

import re

_DITTO_MARKS = {'"', "''", "″", "〃", "”", "“", ",,", "^", "-\"-", "do", "do.", "ditto", "same"}
_NUMBER_LIKE = re.compile(r"[\d.,/| \-]*\d[\d.,/| \-]*")
_DIGITS_WITH_O = re.compile(r"[\dOo.,/| \-]+")


def _is_ditto(cell: str) -> bool:
    compact = re.sub(r"\s+", "", cell).lower()
    return compact in _DITTO_MARKS or (len(compact) > 1 and set(compact) <= {'"'})


def _fill_dittos(rows: list[list[str]]) -> None:
    for c in range(max((len(r) for r in rows), default=0)):
        above = ""
        for row in rows:
            if c >= len(row):
                continue
            if _is_ditto(row[c]):
                if above:
                    row[c] = above
            elif row[c].strip():
                above = row[c]


def _fix_letter_o(rows: list[list[str]]) -> None:
    for c in range(max((len(r) for r in rows), default=0)):
        cells = [row[c].strip() for row in rows if c < len(row) and row[c].strip()]
        has_real_number = any(_NUMBER_LIKE.fullmatch(x) for x in cells)
        numberish = sum(bool(_NUMBER_LIKE.fullmatch(x) or _DIGITS_WITH_O.fullmatch(x)) for x in cells)
        if not has_real_number or numberish / len(cells) < 0.6:
            continue  # not a number column (a lone "O" may be a real letter)
        for row in rows:
            cell = row[c].strip() if c < len(row) else ""
            if cell and "o" in cell.lower() and _DIGITS_WITH_O.fullmatch(cell):
                row[c] = cell.replace("O", "0").replace("o", "0")


def tidy_rows(rows: list[list[str]]) -> list[list[str]]:
    """Return a copy of *rows* (lists of cells) with dittos filled and O->0 fixed."""
    rows = [list(row) for row in rows]
    _fill_dittos(rows)
    _fix_letter_o(rows)
    return rows
