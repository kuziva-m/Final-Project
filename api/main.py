"""
api/main.py — FastAPI backend for digitising handwritten ledger pages.

Endpoints:
    GET  /health   — liveness check (used by Render and the app warm-up)
    POST /scan     — accepts a multipart image + business_id, cleans it,
                     extracts every table on the page, and stores the
                     result as an independent record for that business.
"""

import base64
import os
import sys

import cv2
import numpy as np
from anthropic import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
)
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Allow importing from the repo root (preprocessing package lives there).
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from preprocessing.clean import preprocess  # noqa: E402
from api.extraction import extract_tables  # noqa: E402
from api.database import save_scan  # noqa: E402

app = FastAPI(title="Ledger Digitisation API")

# Allow the mobile app (and any web client) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """Liveness probe. The mobile app polls this to detect cold-start warm-up."""
    return {"status": "ok"}


@app.post("/scan")
async def scan_image(business_id: str = Form(...), file: UploadFile = File(...)):
    """
    Digitise one ledger page for a given business.

    Cleans the image, then detects and extracts every table on the page —
    column headers and rows are read directly from what's written, so no
    fixed schema is assumed since businesses format their ledgers
    differently — and stores the result as an independent record.

    Returns JSON:
        scan_id             — the stored record's id
        cleaned_image        — base64-encoded PNG of the cleaned image
        tables                — [{columns, rows, notes}, ...] as extracted
        overall_confidence   — self-assessed extraction confidence (0-1)
    """
    if not business_id.strip():
        raise HTTPException(status_code=400, detail="business_id is required.")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    cleaned = preprocess(img)

    try:
        result = extract_tables(cleaned)
    except AuthenticationError:
        raise HTTPException(status_code=500, detail="Extraction service is misconfigured.")
    except RateLimitError:
        raise HTTPException(status_code=503, detail="Extraction service is busy — try again shortly.")
    except BadRequestError as e:
        raise HTTPException(status_code=422, detail=f"Could not process image: {e.message}")
    except APIConnectionError:
        raise HTTPException(status_code=502, detail="Could not reach the extraction service.")
    except APIStatusError as e:
        raise HTTPException(status_code=502, detail=f"Extraction service error: {e.message}")

    tables_data = [t.model_dump() for t in result.tables]

    try:
        saved = save_scan(
            business_id=business_id,
            tables=tables_data,
            overall_confidence=result.overall_confidence,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not save scan: {e}")

    _, buf = cv2.imencode(".png", cleaned)
    cleaned_b64 = base64.b64encode(buf).decode("utf-8")

    return {
        "scan_id": saved["id"],
        "cleaned_image": cleaned_b64,
        "tables": tables_data,
        "overall_confidence": result.overall_confidence,
    }
