"""
web/app.py -- minimal Flask front end for the OCR ledger pipeline.

No new OCR/preprocessing/extraction logic lives here; this module only wires
together the existing pipeline:

    preprocessing.clean.preprocess -- deskew/denoise/CLAHE/binarise
    ocr.fusion.fuse                -- Tesseract-on-cleaned + EasyOCR-on-raw,
                                       fused into structured rows

See ocr/fusion.py for the fusion rule: Tesseract's rows are the structural
skeleton (date, item, column layout); EasyOCR supplies clean numeric values
where the two disagree.

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

from extraction.fields import FIELDNAMES
from ocr.fusion import fuse
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

    # Fuse Tesseract-on-cleaned (row structure) with EasyOCR-on-raw (clean
    # values). Passing the already-computed processed_path means fuse() reuses
    # it instead of preprocessing a second time.
    rows = fuse(raw_path, processed_path=processed_path)
    used_engine = "fused (tesseract + easyocr)"

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
    """Download all saved records as CSV or JSON.

    The download is named after the source image when every exported record
    came from the same upload; a flash confirms exactly what was exported
    (visible on the next page view, since a file download doesn't navigate
    the browser away from the current page).
    """
    if fmt not in ("csv", "json"):
        flash(f"Unknown export format '{fmt}'. Use 'csv' or 'json'.")
        return redirect(url_for("index"))

    db = get_db()
    columns = ["id", "date", "item", "qty", "price", "total", "source_file", "created_at"]
    records = [dict(row) for row in db.execute(f"SELECT {', '.join(columns)} FROM records ORDER BY id")]

    sources = {r["source_file"] for r in records if r["source_file"]}
    if len(sources) == 1:
        export_stem = Path(next(iter(sources))).stem
    elif sources:
        export_stem = "records_multiple_sources"
    else:
        export_stem = "records"
    download_name = f"{export_stem}.{fmt}"

    confirmation = f"Exported {len(records)} record(s) as {download_name}"
    if len(sources) == 1:
        confirmation += f" (source: {next(iter(sources))})"
    flash(confirmation + ".")

    if fmt == "csv":
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
        return Response(
            buf.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment; filename={download_name}"},
        )

    import json

    return Response(
        json.dumps(records, indent=2, ensure_ascii=False),
        mimetype="application/json",
        headers={"Content-Disposition": f"attachment; filename={download_name}"},
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
    # 0.0.0.0:7860 matches the Hugging Face Spaces container convention.
    app.run(host="0.0.0.0", port=7860, debug=debug)
