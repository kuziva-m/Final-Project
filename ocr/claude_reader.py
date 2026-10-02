"""
ocr/claude_reader.py -- read a ledger photo with Claude's vision model.

An alternative recognition engine to the Tesseract + EasyOCR fusion in
ocr/fusion.py, with two readers:
    read_tables -- every table with the page's own column headings, since
                   each business formats its records differently
    read_rows   -- the fixed {date, item, qty, price, total} rows
                   (extraction.fields.FIELDNAMES) the mobile API expects

Enabled only when ANTHROPIC_API_KEY is set. Optional overrides:
    LEDGER_CLAUDE_MODEL   (default claude-sonnet-5)
    LEDGER_CLAUDE_EFFORT  (default low -- transcription needs little reasoning, and
                          a live demo needs a fast answer)
"""

from __future__ import annotations

import base64
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

from extraction.fields import FIELDNAMES

DEFAULT_MODEL = "claude-sonnet-5"
# Models documented to accept server-side refusal fallback ("default" mode);
# other models get the plain request.
_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
DEFAULT_EFFORT = "low"

# Claude downsamples anything larger, and the API rejects images over 5 MB,
# which full-resolution phone photos can exceed.
_MAX_EDGE = 1568

_ROWS_PROMPT = """This photo shows a page from a small business's handwritten records \
(a sales book, stock sheet or receipt), possibly skewed, faded or partly in Shona \
or Ndebele.

Find the table on the page and transcribe every transaction row into these fields:
- date: as written (e.g. "12/03")
- item: the item or description, in its original language and spelling
- qty: the quantity as written
- price: the unit price as a plain number (e.g. "2.50"), no currency symbol
- total: the line total as a plain number, no currency symbol

The page's own column headings may differ or be missing; map each column to the \
closest field by its content. Use "" for a field the page doesn't have for that row. \
Copy what is written rather than correcting or calculating it, and put "?" in place \
of characters you cannot read. Skip heading rows and page or column totals. \
Return rows in the order they appear."""

_ROWS_SCHEMA = {
    "type": "object",
    "properties": {
        "rows": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {field: {"type": "string"} for field in FIELDNAMES},
                "required": list(FIELDNAMES),
                "additionalProperties": False,
            },
        }
    },
    "required": ["rows"],
    "additionalProperties": False,
}


_TABLES_PROMPT = """This photo shows a page from a small business's handwritten records. \
Businesses keep very different documents (sales books, stock sheets, debtor lists, \
receipts, cash books), so do not assume a layout: read the structure from the page itself. \
The writing may be skewed or faded and partly in Shona or Ndebele.

For each table on the page:
- title: the table's heading as written, or a short description if it has none
- columns: the column headings exactly as written, left to right. If the page has row \
headings (labels down the left side, such as item names on a stock sheet), make them \
the first column. Give any column without a written heading a short descriptive name.
- rows: every row top to bottom, one cell per column, in the same order as columns.

Copy each cell as written, in its original language and spelling, rather than \
correcting or calculating it. Where a ditto mark means "same as above", write the value \
it stands for. Use "" for an empty cell and "?" for characters you cannot read. Keep \
total and balance rows where they appear on the page. If the page has no table, \
return its lines as a single one-column table."""

_TABLES_SCHEMA = {
    "type": "object",
    "properties": {
        "tables": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "columns": {"type": "array", "items": {"type": "string"}},
                    "rows": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
                },
                "required": ["title", "columns", "rows"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["tables"],
    "additionalProperties": False,
}


class ClaudeReadError(RuntimeError):
    """Claude could not produce rows for this image."""


def is_enabled() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def model_name() -> str:
    return os.environ.get("LEDGER_CLAUDE_MODEL", DEFAULT_MODEL)


def _encode_image(image_path: Path) -> str:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ClaudeReadError(f"Could not read image {image_path.name}.")
    h, w = img.shape[:2]
    scale = _MAX_EDGE / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ClaudeReadError("Could not encode image for upload.")
    return base64.standard_b64encode(buf.tobytes()).decode("ascii")


def _final_text(content) -> str:
    """Text of the answer, ignoring any partial output from a model that was
    replaced by a server-side fallback (marked by a `fallback` block)."""
    parts: list[str] = []
    for block in content:
        if block.type == "fallback":
            parts = []
        elif block.type == "text":
            parts.append(block.text)
    return "".join(parts)


def _ask(image_path: str | os.PathLike, prompt: str, schema: dict) -> dict:
    """Send the photo and *prompt* to Claude; return the JSON answer."""
    import anthropic  # imported lazily so the app runs without the package

    # Extra retries: parallel page reads can briefly hit rate limits, and the
    # SDK backs off and retries those automatically.
    client = anthropic.Anthropic(timeout=90.0, max_retries=4)
    image_b64 = _encode_image(Path(image_path))
    model = model_name()
    fallback = (
        {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
        if model in _FALLBACK_MODELS else {}
    )

    try:
        response = client.beta.messages.create(
            model=model,
            max_tokens=16000,
            **fallback,
            output_config={
                "effort": os.environ.get("LEDGER_CLAUDE_EFFORT", DEFAULT_EFFORT),
                "format": {"type": "json_schema", "schema": schema},
            },
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
    except anthropic.AuthenticationError as exc:
        raise ClaudeReadError("the API key was rejected") from exc
    except anthropic.RateLimitError as exc:
        raise ClaudeReadError("rate limited") from exc
    except anthropic.BadRequestError as exc:
        if "credit" in str(exc.message).lower():
            raise ClaudeReadError("the API account is out of credit") from exc
        raise ClaudeReadError(f"API error 400: {exc.message}") from exc
    except anthropic.APIStatusError as exc:
        raise ClaudeReadError(f"API error {exc.status_code}") from exc
    except anthropic.APIConnectionError as exc:
        raise ClaudeReadError("could not reach the API") from exc

    if response.stop_reason == "refusal":
        raise ClaudeReadError("the model declined this image")
    if response.stop_reason == "max_tokens":
        raise ClaudeReadError("the answer was cut off")

    try:
        return json.loads(_final_text(response.content))
    except json.JSONDecodeError as exc:
        raise ClaudeReadError("the answer was not valid JSON") from exc


def read_rows(image_path: str | os.PathLike) -> list[dict]:
    """Transcribe the ledger in *image_path* into FIELDNAMES rows."""
    rows = _ask(image_path, _ROWS_PROMPT, _ROWS_SCHEMA).get("rows", [])
    return [{field: str(row.get(field, "")).strip() for field in FIELDNAMES} for row in rows]


def read_tables(image_path: str | os.PathLike) -> list[dict]:
    """Transcribe every table on the page using the page's own headings.

    Returns ``[{"title": str, "columns": [str, ...], "rows": [[str, ...], ...]}]``
    with every row padded or trimmed to the number of columns.
    """
    tables = []
    for table in _ask(image_path, _TABLES_PROMPT, _TABLES_SCHEMA).get("tables", []):
        columns = [str(c).strip() for c in table.get("columns", [])]
        if not columns:
            continue
        width = len(columns)
        rows = [
            ([str(cell).strip() for cell in row] + [""] * width)[:width]
            for row in table.get("rows", [])
        ]
        tables.append({"title": str(table.get("title", "")).strip(), "columns": columns, "rows": rows})
    return tables


# ---------------------------------------------------------------------------
# Multi-page documents (PDF scans): one request per page, a few at a time.
# ---------------------------------------------------------------------------


def _workers() -> int:
    try:
        return max(1, int(os.environ.get("LEDGER_PDF_WORKERS", "3")))
    except ValueError:
        return 3


def _try(read, image_path):
    try:
        return read(image_path)
    except ClaudeReadError as exc:
        return exc


def read_rows_from_pages(image_paths: list[Path]) -> list[dict]:
    """FIELDNAMES rows from every page, in page order. Any failed page raises,
    so the caller can fall back for the whole document."""
    with ThreadPoolExecutor(max_workers=_workers()) as pool:
        results = list(pool.map(lambda p: _try(read_rows, p), image_paths))
    for result in results:
        if isinstance(result, ClaudeReadError):
            raise result
    return [row for page_rows in results for row in page_rows]


def _same_headings(a: list[str], b: list[str]) -> bool:
    return [c.strip().lower() for c in a] == [c.strip().lower() for c in b]


def merge_continued_tables(page_tables: list[dict], multi_page: bool) -> list[dict]:
    """Join a table that carries on from the end of one page onto the next.

    *page_tables* is in page order, each with a ``page`` number. A table is
    joined to the one before it when it starts the very next page, has the
    same headings, and its title is blank or the same. On multi-page
    documents each title is labelled with the page(s) it came from.
    """
    merged: list[dict] = []
    for table in page_tables:
        prev = merged[-1] if merged else None
        if (
            prev is not None
            and table["page"] == prev["pages"][-1] + 1
            and _same_headings(prev["columns"], table["columns"])
            and table["title"].strip().lower() in ("", prev["title"].strip().lower())
        ):
            prev["rows"].extend(table["rows"])
            prev["pages"].append(table["page"])
            continue
        merged.append({"title": table["title"], "columns": table["columns"],
                       "rows": list(table["rows"]), "pages": [table["page"]]})

    for table in merged:
        pages = table.pop("pages")
        if multi_page:
            label = f"page {pages[0]}" if len(pages) == 1 else f"pages {pages[0]}\u2013{pages[-1]}"
            table["title"] = f"{table['title']} ({label})" if table["title"] else label.capitalize()
    return merged


def read_tables_from_pages(image_paths: list[Path]) -> list[dict]:
    """Tables from every page, with continued tables joined across pages.

    A page that fails becomes an empty table whose title says why; if every
    page fails, the first error is raised so the caller can fall back.
    """
    with ThreadPoolExecutor(max_workers=_workers()) as pool:
        results = list(pool.map(lambda p: _try(read_tables, p), image_paths))
    if all(isinstance(r, ClaudeReadError) for r in results):
        raise results[0]

    page_tables = []
    for page, result in enumerate(results, start=1):
        if isinstance(result, ClaudeReadError):
            page_tables.append({"title": f"Could not be read: {result}", "columns": ["Column 1"],
                                "rows": [[""]], "page": page})
        else:
            page_tables.extend({**table, "page": page} for table in result)
    return merge_continued_tables(page_tables, multi_page=len(image_paths) > 1)
