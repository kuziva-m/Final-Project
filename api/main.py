"""
api/main.py — FastAPI backend wrapping the OCR preprocessing pipeline.

Endpoints:
    GET  /health        — liveness check (used by Render and the app warm-up)
    POST /process       — accepts a multipart image, returns cleaned image + OCR text
"""

import base64
import os
import sys

import cv2
import easyocr
import numpy as np
import pytesseract
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Allow importing from the repo root (preprocessing package lives there).
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from preprocessing.clean import preprocess  # noqa: E402

app = FastAPI(title="Ledger OCR API")

# Allow the Flutter app (and any web client) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build the EasyOCR reader once at startup — model download is ~100 MB and
# is baked into the Docker image (see Dockerfile), so this is fast on boot.
_easyocr_reader = easyocr.Reader(["en"], gpu=False)


@app.get("/health")
def health():
    """Liveness probe. The Flutter app polls this to detect cold-start warm-up."""
    return {"status": "ok"}


@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    """
    Accept an image upload, clean it, and run both OCR engines.

    Returns JSON:
        cleaned_image  — base64-encoded PNG of the cleaned image
        tesseract_text — Tesseract OCR output string
        easyocr_text   — EasyOCR output string
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    cleaned = preprocess(img)

    _, buf = cv2.imencode(".png", cleaned)
    cleaned_b64 = base64.b64encode(buf).decode("utf-8")

    tess_text = pytesseract.image_to_string(Image.fromarray(cleaned)).strip()
    easy_results = _easyocr_reader.readtext(cleaned, detail=0, paragraph=False)
    easy_text = "\n".join(easy_results).strip()

    return {
        "cleaned_image": cleaned_b64,
        "tesseract_text": tess_text,
        "easyocr_text": easy_text,
    }
