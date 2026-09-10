"""
api/extraction.py — vision-based table detection and structured extraction.

Reads whatever table structure is on the page (column headers vary by
business) and returns it as columns + rows, so no fixed ledger schema is
assumed anywhere in the pipeline.
"""

import base64
import os
from typing import List

import cv2
from anthropic import Anthropic
from pydantic import BaseModel

# Vision-language model used for table detection + extraction. Configurable
# via env var so it can be swapped without a code change.
MODEL_ID = os.environ.get("EXTRACTION_MODEL_ID", "claude-sonnet-5")

_EXTRACTION_INSTRUCTIONS = """\
This image is a handwritten business ledger page. Find every table on the \
page. For each table:
- Read the column headers exactly as written. If the table has no header \
row, infer short column names from context (e.g. "Date", "Item").
- Extract every row as a list of cell values, in the same order as the \
columns, reading top to bottom.
- Keep numbers exactly as written — do not reformat dates or currency.
- If a cell is illegible, write "?" for that cell and describe it in notes.
- If there is nothing to note, set notes to an empty string.
- If no table is visible on the page, return an empty tables list.

Also give an overall_confidence between 0.0 and 1.0 for how certain you are \
about the extraction as a whole — lower it whenever handwriting is unclear \
or cells were guessed."""


class ExtractedTable(BaseModel):
    columns: List[str]
    rows: List[List[str]]
    notes: str


class ExtractionResult(BaseModel):
    tables: List[ExtractedTable]
    overall_confidence: float


_client = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    return _client


def extract_tables(image) -> ExtractionResult:
    """
    Detect and extract every table in a cleaned ledger image.

    `image` is a numpy array (grayscale or BGR) — typically the output of
    preprocessing.clean.preprocess(). Column headers and row values are read
    directly from the page, so this works across businesses without any
    fixed ledger schema.
    """
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Could not encode image for extraction.")
    image_b64 = base64.standard_b64encode(buf).decode("utf-8")

    client = _get_client()
    response = client.messages.parse(
        model=MODEL_ID,
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_b64,
                    },
                },
                {"type": "text", "text": _EXTRACTION_INSTRUCTIONS},
            ],
        }],
        output_format=ExtractionResult,
    )

    return response.parsed_output
