"""
web/app.py -- minimal Flask front end for the OCR ledger pipeline.

No new OCR/preprocessing/extraction logic lives here; this module only wires
together the existing pipeline:

    preprocessing.clean.preprocess   -- deskew/denoise/CLAHE/binarise
    ocr.compare.run_easyocr          -- on the raw upload
    ocr.compare.run_tesseract        -- on the preprocessed upload
    extraction.fields.extract_fields -- raw OCR text -> structured rows

Pipeline rule (see ocr/compare.py): Tesseract reads the preprocessed image
better (deskew fixes its dropped decimals / 5-vs-S errors); EasyOCR is already
tolerant of raw skew. Extraction tries the Tesseract-on-cleaned text first
since it is the stronger source for structured numbers, falling back to
EasyOCR-on-raw only if Tesseract yields no parseable rows.

Run:
    python -m web.app
    (or) python web/app.py
"""

from __future__ import annotations

import csv
import io
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow `python web/app.py` to find the top-level packages (preprocessing,
# ocr, extraction) regardless of the process's working directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
from flask import (
    Flask,
    Response,
    flash,
    g,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from werkzeug.utils import secure_filename

from extraction.fields import FIELDNAMES, extract_fields
from ocr.compare import run_easyocr, run_tesseract
from preprocessing.clean import preprocess

# ---------------------------------------------------------------------------
# Paths -- everything reads from / writes to folders, nothing is hardcoded.
# ---------------------------------------------------------------------------

DATA_DIR = _REPO_ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
DB_PATH = DATA_DIR / "records.db"

for _dir in (UPLOADS_DIR, PROCESSED_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-in-production")


# ---------------------------------------------------------------------------
# SQLite -- stdlib sqlite3, table auto-created on startup.
# ---------------------------------------------------------------------------


def init_db() -> None:
    """Create the records table if it does not already exist."""
    with sqlite3.connect(str(DB_PATH)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS records (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                date        TEXT,
                item        TEXT,
                qty         TEXT,
                price       TEXT,
                total       TEXT,
                source_file TEXT,
                created_at  TEXT NOT NULL
            )
            """
        )
        conn.commit()


def get_db() -> sqlite3.Connection:
    """Return the request-scoped SQLite connection, creating it if needed."""
    if "db" not in g:
        g.db = sqlite3.connect(str(DB_PATH))
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_exc: BaseException | None) -> None:
    db = g.pop("db", None)
    if db is not None:
        db.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _unique_stem(original_filename: str) -> str:
    """Timestamp-prefixed, path-safe stem so repeat uploads never collide."""
    safe_name = secure_filename(original_filename)
    stem = Path(safe_name).stem or "upload"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    return f"{timestamp}_{stem}"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    """Upload form for a single ledger image."""
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    """Save the upload, run the pipeline, and show an editable rows table."""
    upload = request.files.get("image")

    if upload is None or upload.filename == "":
        flash("No file selected. Choose an image and try again.")
        return redirect(url_for("index"))

    if not _allowed_file(upload.filename):
        flash(
            f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
        return redirect(url_for("index"))

    stem = _unique_stem(upload.filename)
    ext = Path(secure_filename(upload.filename)).suffix.lower()
    raw_filename = f"{stem}{ext}"
    raw_path = UPLOADS_DIR / raw_filename
    upload.save(str(raw_path))

    img = cv2.imread(str(raw_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raw_path.unlink(missing_ok=True)
        flash("Could not read that image. It may be corrupt or an unsupported format.")
        return redirect(url_for("index"))

    processed_filename = f"{stem}_clean.png"
    processed_path = PROCESSED_DIR / processed_filename
    preprocess(img, save_path=processed_path)

    # Pipeline rule: Tesseract on the cleaned image, EasyOCR on the raw image.
    tesseract_text = run_tesseract(processed_path)
    easyocr_text = run_easyocr(raw_path)

    rows = extract_fields(tesseract_text)
    used_engine = "tesseract"
    if not rows:
        rows = extract_fields(easyocr_text)
        used_engine = "easyocr"

    if not rows:
        # Never dead-end the owner with nothing to correct -- give one blank row.
        rows = [{k: "" for k in FIELDNAMES}]
        used_engine = "none (manual entry)"

    return render_template(
        "process.html",
        rows=rows,
        source_file=raw_filename,
        raw_filename=raw_filename,
        processed_filename=processed_filename,
        used_engine=used_engine,
        tesseract_text=tesseract_text,
        easyocr_text=easyocr_text,
    )


@app.route("/save", methods=["POST"])
def save():
    """Persist the (possibly corrected) rows from /process into SQLite."""
    source_file = request.form.get("source_file", "").strip()
    try:
        num_rows = int(request.form.get("num_rows", "0"))
    except ValueError:
        num_rows = 0

    created_at = datetime.now(timezone.utc).isoformat()
    saved = 0

    db = get_db()
    for i in range(num_rows):
        row = {field: request.form.get(f"{field}_{i}", "").strip() for field in FIELDNAMES}
        if not any(row.values()):
            continue  # skip fully-blank rows (e.g. an unused manual-entry row)
        db.execute(
            """
            INSERT INTO records (date, item, qty, price, total, source_file, created_at)
            VALUES (:date, :item, :qty, :price, :total, :source_file, :created_at)
            """,
            {**row, "source_file": source_file, "created_at": created_at},
        )
        saved += 1
    db.commit()

    flash(f"Saved {saved} record(s) from {source_file or 'upload'}.")
    return redirect(url_for("index"))


@app.route("/export/<fmt>")
def export(fmt: str):
    """Download all saved records as CSV or JSON."""
    if fmt not in ("csv", "json"):
        flash(f"Unknown export format '{fmt}'. Use 'csv' or 'json'.")
        return redirect(url_for("index"))

    db = get_db()
    columns = ["id", "date", "item", "qty", "price", "total", "source_file", "created_at"]
    records = [dict(row) for row in db.execute(f"SELECT {', '.join(columns)} FROM records ORDER BY id")]

    if fmt == "csv":
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
        return Response(
            buf.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=records.csv"},
        )

    import json

    return Response(
        json.dumps(records, indent=2, ensure_ascii=False),
        mimetype="application/json",
        headers={"Content-Disposition": "attachment; filename=records.json"},
    )


@app.route("/uploads/<path:filename>")
def uploaded_file(filename: str):
    """Serve a raw upload for preview in the /process page."""
    return send_from_directory(str(UPLOADS_DIR), filename)


@app.route("/processed/<path:filename>")
def processed_file(filename: str):
    """Serve a preprocessed image for preview in the /process page."""
    return send_from_directory(str(PROCESSED_DIR), filename)


init_db()


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="127.0.0.1", port=5000, debug=debug)
