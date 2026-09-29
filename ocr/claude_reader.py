"""
ocr/claude_reader.py -- read a ledger photo with Claude's vision model.

An alternative recognition engine to the Tesseract + EasyOCR fusion in
ocr/fusion.py. It returns rows in the same {date, item, qty, price, total}
shape (extraction.fields.FIELDNAMES), so review, storage and export are
unchanged whichever engine produced the rows.

Enabled only when ANTHROPIC_API_KEY is set. Optional overrides:
    LEDGER_CLAUDE_MODEL   (default claude-opus-5-5)
    LEDGER_CLAUDE_EFFORT  (default low -- transcription needs little reasoning, and
                          a live demo needs a fast answer)
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import cv2

from extraction.fields import FIELDNAMES

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "low"

# Claude downsamples anything larger, and the API rejects images over 5 MB,
# which full-resolution phone photos can exceed.
_MAX_EDGE = 1568

_PROMPT = """This photo shows a page from a small business's handwritten records \
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

_SCHEMA = {
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


def read_rows(image_path: str | os.PathLike) -> list[dict]:
    """Transcribe the ledger in *image_path* into FIELDNAMES rows."""
    import anthropic  # imported lazily so the app runs without the package

    client = anthropic.Anthropic(timeout=90.0, max_retries=2)
    image_b64 = _encode_image(Path(image_path))

    try:
        response = client.beta.messages.create(
            model=model_name(),
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={
                "effort": os.environ.get("LEDGER_CLAUDE_EFFORT", DEFAULT_EFFORT),
                "format": {"type": "json_schema", "schema": _SCHEMA},
            },
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}},
                    {"type": "text", "text": _PROMPT},
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
        rows = json.loads(_final_text(response.content))["rows"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ClaudeReadError("the answer was not valid JSON") from exc

    return [{field: str(row.get(field, "")).strip() for field in FIELDNAMES} for row in rows]
