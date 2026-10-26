"""
web/statement.py -- a saved document laid out like a bank statement.

Lenders judge a small business by its cash flow, which they read from bank
statements: date, description, reference, money out (debit), money in
(credit) and a running balance, between an opening and a closing balance.
This lays the tables of a saved document out the same way, so records kept
on paper can sit beside the business's bank statements in a loan
application, or be imported by tools that read statements.

It is made from the owner's own records, and every copy says so: it is not
a statement issued by a bank.

Each column of the document has a role (ROLES). guess_roles() suggests one
from the heading and the cells; the owner can change it on the page.
"""

from __future__ import annotations

import csv
import io
import re
from datetime import datetime

from web.exports import _parse_date

ROLES = ("date", "description", "reference", "in", "out", "signed", "skip")
CURRENCIES = ("USD", "ZiG", "ZAR")
TITLE = "Statement of business transactions"
NOT_A_BANK = "Prepared from the business's own records with MSME Ledger. Not issued by a bank."
COLUMNS = ["Date", "Description", "Reference", "Debit (money out)", "Credit (money in)", "Balance"]

_DATE_HEADING = re.compile(r"\b(date|day|zuva|usuku)\b", re.I)
_IN = re.compile(r"\b(money in|cash in|in|income|sales?|sold|takings|receipts?|revenue|credits?|deposits?)\b", re.I)
_OUT = re.compile(r"\b(money out|cash out|out|expenses?|expenditure|paid|payments?|purchases?|bought|spent|costs?|"
                  r"debits?|withdrawals?)\b", re.I)
_AMOUNT = re.compile(r"\b(amount|total|value|sum|mari)\b", re.I)
_REFERENCE = re.compile(r"\b(ref|reference|receipt no|invoice|voucher|cheque)\b", re.I)
_ISO_DATE = re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})")
_CURRENCY_MARKS = re.compile(r"(?i)us\$|usd|zig|zwg|zar|rtgs|\$|£|€|\bR(?=\s*\d)")
_GROUPED = re.compile(r"-?\d{1,3}(?:[ ,]\d{3})+(?:\.\d+)?")
_PLAIN = re.compile(r"-?\d+(?:\.\d+)?")
_DECIMAL_COMMA = re.compile(r"-?\d+,\d{1,2}")
_LEADING_ZERO = re.compile(r"-?0\d")  # "001", "0123": a code or number, not an amount
# Marks meaning "same as the line above" in a handwritten book.
_DITTO = {'"', "''", "〃", "„", "“", "”", "do", "do.", "-do-", "ditto"}


def heading_key(heading: str) -> str:
    """Headings that differ only in case or spacing are the same column."""
    return " ".join(heading.lower().split())


def parse_date(text: str) -> datetime | None:
    """Day-first dates as written in a ledger (29.11.17, 20|12|17), or 2026-01-05."""
    value = text.strip()
    match = _ISO_DATE.fullmatch(value)
    if match:
        try:
            return datetime(*(int(part) for part in match.groups()))
        except ValueError:
            return None
    return _parse_date(value)


def parse_amount(text: str) -> float | None:
    """'15 000', '5,000', '$4.25', '(20)' or '1,15' as a number; None if it isn't one."""
    value = _CURRENCY_MARKS.sub("", text).replace("−", "-").strip()
    negative = value.startswith("(") and value.endswith(")")
    value = value.strip("()").strip()
    if _LEADING_ZERO.match(value):
        return None
    if _GROUPED.fullmatch(value):
        number = float(re.sub(r"[ ,]", "", value))
    elif _PLAIN.fullmatch(value):
        number = float(value)
    elif _DECIMAL_COMMA.fullmatch(value):  # "1,15": the reader's comma for a decimal point
        number = float(value.replace(",", "."))
    else:
        return None
    return -number if negative else number


def _undittoed(rows: list[list[str]]) -> list[list[str]]:
    """Rows with each ditto mark replaced by the value above it."""
    out: list[list[str]] = []
    for row in rows:
        filled = []
        for c, cell in enumerate(row):
            if cell.strip().lower() in _DITTO and out and c < len(out[-1]):
                cell = out[-1][c]
            filled.append(cell)
        out.append(filled)
    return out


def guess_roles(columns: list[str], rows: list[list[str]]) -> list[str]:
    """A suggested role for each column, from its heading and what is in it."""
    rows = _undittoed(rows)

    def share(c: int, test) -> float:
        cells = [r[c].strip() for r in rows if c < len(r) and r[c].strip()]
        return sum(test(x) for x in cells) / len(cells) if cells else 0.0

    roles = ["reference" if _REFERENCE.search(heading) else "skip" for heading in columns]
    dates = [c for c in range(len(columns))
             if roles[c] == "skip" and share(c, lambda x: parse_date(x) is not None) >= 0.6]
    dates.sort(key=lambda c: not _DATE_HEADING.search(columns[c]))  # a "Date" heading first
    if dates:
        roles[dates[0]] = "date"
    money = [c for c in range(len(columns))
             if roles[c] == "skip" and share(c, lambda x: parse_amount(x) is not None) >= 0.6]
    for c in money:
        if _OUT.search(columns[c]):
            roles[c] = "out"
        elif _IN.search(columns[c]):
            roles[c] = "in"
    if not any(r in ("in", "out") for r in roles):
        # One amount column and no direction: a sales book, most likely.
        amounts = [c for c in money if _AMOUNT.search(columns[c])]
        if amounts:
            roles[amounts[-1]] = "in"
    for c in range(len(columns)):
        if roles[c] == "skip" and c not in money and share(c, lambda x: parse_date(x) is None) >= 0.6:
            roles[c] = "description"
            break
    return roles


def document_columns(tables: list[dict]) -> list[dict]:
    """Each distinct column heading of a document, in order, with example cells
    and a suggested role: [{"key", "heading", "examples", "guess"}]."""
    found: dict[str, dict] = {}
    for table in tables:
        guesses = guess_roles(table["columns"], table["rows"])
        for c, heading in enumerate(table["columns"]):
            entry = found.setdefault(heading_key(heading), {
                "key": heading_key(heading), "heading": heading, "examples": [], "guess": guesses[c]})
            for row in table["rows"]:
                if len(entry["examples"]) >= 3:
                    break
                cell = row[c].strip() if c < len(row) else ""
                if cell and cell.lower() not in _DITTO and cell not in entry["examples"]:
                    entry["examples"].append(cell)
    return list(found.values())


def build(tables: list[dict], roles: dict[str, str], opening: float = 0.0) -> dict:
    """The statement: its lines with running balances, and the totals.

    Rows without an amount in a money column are left out (and counted). Lines
    are put in date order when every line has a date; otherwise the book's
    order is kept. A row with no date of its own takes the date above it.
    """
    lines, left_out = [], 0
    for table in sorted(tables, key=lambda t: t.get("page") or 1):
        col_roles = [roles.get(heading_key(h), "skip") for h in table["columns"]]
        last_date = None
        for row in _undittoed(table["rows"]):
            if not any(cell.strip() for cell in row):
                continue
            cells = {role: [] for role in ROLES}
            for c, role in enumerate(col_roles):
                if c < len(row) and row[c].strip():
                    cells[role].append(row[c].strip())
            money_in = money_out = 0.0
            for role in ("in", "out", "signed"):
                for text in cells[role]:
                    amount = parse_amount(text)
                    if amount is None:
                        continue
                    if role == "out" or (role == "signed" and amount < 0):
                        money_out += abs(amount)
                    else:
                        money_in += abs(amount)
            date = next((d for d in map(parse_date, cells["date"]) if d), None) or last_date
            last_date = date
            if not money_in and not money_out:
                left_out += 1
                continue
            lines.append({"date": date, "description": " · ".join(cells["description"]),
                          "reference": " · ".join(cells["reference"]),
                          "out": round(money_out, 2), "in": round(money_in, 2)})
    if lines and all(line["date"] for line in lines):
        lines.sort(key=lambda line: line["date"])
    balance = opening
    for line in lines:
        balance = round(balance + line["in"] - line["out"], 2)
        line["balance"] = balance
    dates = [line["date"] for line in lines if line["date"]]
    return {
        "lines": lines,
        "opening": round(opening, 2),
        "closing": round(balance, 2),
        "total_in": round(sum(line["in"] for line in lines), 2),
        "total_out": round(sum(line["out"] for line in lines), 2),
        "start": min(dates) if dates else None,
        "end": max(dates) if dates else None,
        "left_out": left_out,
    }


def _day(value: datetime | None) -> str:
    return value.strftime("%d/%m/%Y") if value else ""


def period(statement: dict) -> str:
    if not statement["start"]:
        return ""
    return f"{_day(statement['start'])} to {_day(statement['end'])}"


def csv_text(statement: dict) -> str:
    """One line per transaction, in the column order statement readers expect."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Date", "Description", "Reference", "Debit", "Credit", "Balance"])
    money = lambda v: f"{v:.2f}" if v else ""  # noqa: E731
    for line in statement["lines"]:
        writer.writerow([_day(line["date"]), line["description"], line["reference"],
                         money(line["out"]), money(line["in"]), f"{line['balance']:.2f}"])
    return buf.getvalue()


def workbook(statement: dict, business: str, currency: str, prepared: datetime) -> bytes:
    """An Excel sheet laid out like a printed statement: details, totals, then the lines."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

    wb = Workbook()
    ws = wb.active
    ws.title = "Statement"
    money_format = '#,##0.00;-#,##0.00;""'
    ws.append([TITLE])
    ws["A1"].font = Font(bold=True, size=14)
    ws.append([NOT_A_BANK])
    ws["A2"].font = Font(italic=True, color="6E6A78")
    ws.append([])
    for label, value in (("Business", business), ("Currency", currency), ("Period", period(statement)),
                         ("Prepared on", _day(prepared)), ("Transactions", len(statement["lines"]))):
        ws.append([label, value])
        ws.cell(row=ws.max_row, column=1).font = Font(bold=True)
    ws.append([])
    for label, value in (("Opening balance", statement["opening"]), ("Total money in (credits)", statement["total_in"]),
                         ("Total money out (debits)", statement["total_out"]), ("Closing balance", statement["closing"])):
        ws.append([label, value])
        ws.cell(row=ws.max_row, column=1).font = Font(bold=True)
        ws.cell(row=ws.max_row, column=2).number_format = "#,##0.00"
    ws.append([])

    ws.append(COLUMNS)
    header = ws.max_row
    for cell in ws[header]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="0B1040")
    ws.append([None, "Opening balance", None, None, None, statement["opening"]])
    for line in statement["lines"]:
        ws.append([line["date"], line["description"], line["reference"], line["out"] or None, line["in"] or None,
                   line["balance"]])
    ws.append([None, "Closing balance", None, statement["total_out"], statement["total_in"], statement["closing"]])
    last = ws.max_row
    for row in ws.iter_rows(min_row=header + 1, max_row=last):
        row[0].number_format = "DD/MM/YYYY"
        for cell in row[3:6]:
            cell.number_format = money_format
            cell.alignment = Alignment(horizontal="right")
    for cell in ws[last]:
        cell.font = Font(bold=True)
        cell.border = Border(top=Side(style="thin"))
    ws.freeze_panes = f"A{header + 1}"
    for letter, width in zip("ABCDEF", (26, 44, 16, 18, 18, 16)):  # A also holds the labels above
        ws.column_dimensions[letter].width = width
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
