"""
web/app.py -- Flask front end for the OCR ledger pipeline, serving both the
HTML review UI and a JSON API for the companion mobile app.

No new OCR/preprocessing/extraction logic lives here; this module only wires
together the existing pipeline:

    preprocessing.clean.preprocess -- deskew/denoise/CLAHE/binarise
    ocr.fusion.run_engines         -- Tesseract-on-cleaned + EasyOCR-on-raw
    ocr.fusion.fuse_from_texts     -- fused into structured rows + provenance

See ocr/fusion.py for the fusion rule: Tesseract's rows are the structural
skeleton (date, item, column layout); EasyOCR supplies clean numeric values
where the two disagree, and every decision is logged (provenance) so the
result is auditable -- no model, only rules.

When ANTHROPIC_API_KEY is set, ocr/claude_reader.py reads the photo instead
and returns rows in the same shape; if that call fails, the fusion above is
used. The engine that produced the rows is always reported as used_engine.

Documents read with their own headings are saved in the scans table and get
their own page (/documents/<id>) to download, share, edit or add pages to.

Run:
    python -m web.app
    (or) python web/app.py
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import sqlite3
import sys
import threading
import time
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
    flash,
    g,
    has_request_context,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    send_from_directory,
    url_for,
)
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from extraction.fields import FIELDNAMES
from ocr import claude_reader
from ocr.fusion import fuse_from_texts, run_engines
from preprocessing.clean import preprocess
from preprocessing.pdf_pages import PdfError, render_pages
from web.exports import build_workbook, document_label
from web.i18n import JS_STRINGS, LANGUAGES, translate

# ---------------------------------------------------------------------------
# Paths -- everything reads from / writes to folders, nothing is hardcoded.
# ---------------------------------------------------------------------------

DATA_DIR = _REPO_ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
DB_PATH = DATA_DIR / "records.db"

for _dir in (UPLOADS_DIR, PROCESSED_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".pdf"}

app = Flask(__name__)
# Scanned PDFs can be large; reject anything over the limit with a clear message.
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("LEDGER_MAX_UPLOAD_MB", "100")) * 1024 * 1024
# Flask's 500 KB default for submitted form fields is too small for a review
# page holding many pages of tables.
app.config["MAX_FORM_MEMORY_SIZE"] = 16 * 1024 * 1024
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-in-production")


# ---------------------------------------------------------------------------
# Language -- English, chiShona or isiNdebele, chosen with a cookie. Strings
# are keyed by their English text (see web/i18n.py); anything without a
# translation shows in English.
# ---------------------------------------------------------------------------


def current_lang() -> str:
    lang = request.cookies.get("lang", "en") if has_request_context() else "en"
    return lang if lang in LANGUAGES else "en"


def tr(text: str, **values) -> str:
    """*text* in the visitor's language, with ``{name}`` placeholders filled."""
    return translate(text, current_lang(), **values)


app.jinja_env.globals.update(_=tr, LANGUAGES=LANGUAGES)


@app.context_processor
def inject_language() -> dict:
    return {"lang": current_lang(), "js_strings": {text: tr(text) for text in JS_STRINGS}}


# ---------------------------------------------------------------------------
# SQLite -- stdlib sqlite3, table auto-created on startup.
# ---------------------------------------------------------------------------


def init_db() -> None:
    """Create the tables if they do not already exist.

    records -- fixed date/item/qty/price/total rows (OCR fusion, mobile app).
    scans   -- documents read with their own headings; tables_json holds
               [{"title", "columns", "rows", "page", "image"}], since columns
               vary per business. name is the owner's name for the document.
    """
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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scans (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                source_file TEXT,
                engine      TEXT,
                tables_json TEXT NOT NULL,
                created_at  TEXT NOT NULL
            )
            """
        )
        # Columns added after the first release; older databases get them here.
        existing = {row[1] for row in conn.execute("PRAGMA table_info(scans)")}
        for column in ("name", "updated_at"):
            if column not in existing:
                conn.execute(f"ALTER TABLE scans ADD COLUMN {column} TEXT")
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


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, default)))
    except ValueError:
        return default


# A scanned record book can be dozens of pages; each page read by Claude
# costs a few cents, so cap the pages read per PDF. The free OCR fallback
# takes far longer per page on CPU, so it reads fewer.
MAX_PDF_PAGES = _env_int("LEDGER_MAX_PDF_PAGES", 40)
MAX_FALLBACK_PDF_PAGES = _env_int("LEDGER_MAX_FALLBACK_PDF_PAGES", 5)


def _pages_note(read: int, total: int, is_pdf: bool) -> str:
    if not is_pdf:
        return ""
    if read < total:
        return tr("first {read} of {total} pages", read=read, total=total)
    return tr("1 page") if total == 1 else tr("{n} pages", n=total)


def _save_upload(upload) -> tuple[Path | None, str | None]:
    """Validate and save *upload*; return ``(saved_path, None)`` or ``(None, error)``."""
    if upload is None or upload.filename == "":
        return None, tr("No file selected. Choose a photo or PDF and try again.")

    if not _allowed_file(upload.filename):
        return None, tr("Unsupported file type. Allowed: {types}", types=", ".join(sorted(ALLOWED_EXTENSIONS)))

    stem = _unique_stem(upload.filename)
    ext = Path(secure_filename(upload.filename)).suffix.lower()
    source_path = UPLOADS_DIR / f"{stem}{ext}"
    upload.save(str(source_path))
    return source_path, None


def _uploaded_path(token: str) -> Path | None:
    """The saved upload named by *token* (from /upload), or None if invalid."""
    name = secure_filename(token or "")
    if not name or name != token or not _allowed_file(name):
        return None
    path = UPLOADS_DIR / name
    return path if path.is_file() else None


# Light JPEGs of each page for the browser: the full renders are multi-MB
# PNGs, too heavy for phones on mobile data.
_VIEW_EDGE, _THUMB_EDGE = 1400, 240


def _page_images(page_path: Path) -> dict:
    """Write a page's viewing JPEG and thumbnail; return their file names."""
    img = cv2.imread(str(page_path), cv2.IMREAD_COLOR)
    if img is None:
        return {}
    names = {}
    for kind, edge, quality in (("view", _VIEW_EDGE, 80), ("thumb", _THUMB_EDGE, 70)):
        h, w = img.shape[:2]
        scale = min(1.0, edge / max(h, w))
        small = img if scale >= 1.0 else cv2.resize(
            img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
        name = f"{page_path.stem}_{kind}.jpg"
        cv2.imwrite(str(PROCESSED_DIR / name), small, [cv2.IMWRITE_JPEG_QUALITY, quality])
        names[kind] = name
    return names


def _thumb_for(view_name: str | None) -> str | None:
    if view_name and view_name.endswith("_view.jpg"):
        thumb = view_name[: -len("_view.jpg")] + "_thumb.jpg"
        if (PROCESSED_DIR / thumb).is_file():
            return thumb
    return None


# ---------------------------------------------------------------------------
# Reading progress -- which pages of an upload are done, for the waiting
# screen. Kept in memory: the Space runs a single server process.
# ---------------------------------------------------------------------------

_progress: dict[str, dict] = {}
_progress_lock = threading.Lock()
_PROGRESS_TTL_SECONDS = 3600


def _progress_update(token: str | None, **fields) -> None:
    if not token:
        return
    now = time.time()
    with _progress_lock:
        for key in [k for k, v in _progress.items() if now - v["updated"] > _PROGRESS_TTL_SECONDS]:
            del _progress[key]
        entry = _progress.setdefault(token, {"stage": "preparing", "total": 0, "done": set(), "thumbs": []})
        entry.update(fields)
        entry["updated"] = now


def _progress_page_done(token: str | None, index: int) -> None:
    if not token:
        return
    with _progress_lock:
        entry = _progress.get(token)
        if entry is not None:
            entry["done"].add(index)
            entry["updated"] = time.time()


def _run_pipeline_on_upload(upload, detect_tables: bool = False) -> tuple[dict | None, str | None]:
    """Save *upload*, then run the pipeline on it (see _run_pipeline_on_file)."""
    source_path, error = _save_upload(upload)
    if error:
        return None, error
    return _run_pipeline_on_file(source_path, detect_tables)


def _run_pipeline_on_file(source_path: Path, detect_tables: bool = False,
                          token: str | None = None) -> tuple[dict | None, str | None]:
    """Run preprocess + recognition on a saved upload and return the result.

    Shared by the HTML /process route and the JSON /api/scan route so both
    surfaces run the exact same pipeline. Returns ``(artifacts, None)`` on
    success or ``(None, error_message)`` on a validation/decode failure.

    A PDF is rendered to one image per page first; every page then goes
    through the same per-image pipeline, and the first page is the preview.
    With a *token*, progress is reported page by page (see /progress).

    ``artifacts`` keys: used_engine, fallback_reason, source_file,
    raw_filename, processed_filename, pages_note, pages (each page's
    viewing JPEG and thumbnail), plus either ``tables`` (when *detect_tables*
    and Claude read the pages with their own headings) or ``rows`` +
    ``provenance`` (fixed FIELDNAMES rows).
    """
    source_file = source_path.name
    stem = source_path.stem
    ext = source_path.suffix.lower()
    _progress_update(token, stage="preparing")

    is_pdf = ext == ".pdf"
    if is_pdf:
        try:
            page_paths, total_pages = render_pages(source_path, UPLOADS_DIR, stem, MAX_PDF_PAGES)
        except PdfError as exc:
            source_path.unlink(missing_ok=True)
            return None, tr(str(exc))
    else:
        page_paths, total_pages = [source_path], 1

    def clean_page(page_path: Path) -> Path | None:
        img = cv2.imread(str(page_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        clean_path = PROCESSED_DIR / f"{page_path.stem}_clean.png"
        preprocess(img, save_path=clean_path)
        return clean_path

    first_clean = clean_page(page_paths[0])
    if first_clean is None:
        source_path.unlink(missing_ok=True)
        return None, tr("Could not read that image. It may be corrupt or an unsupported format.")

    pages = [{"page": n, **_page_images(path)} for n, path in enumerate(page_paths, start=1)]
    files = {
        "source_file": source_file,
        "raw_filename": page_paths[0].name,
        "processed_filename": first_clean.name,
        "pages": pages,
    }
    _progress_update(token, stage="reading", total=len(page_paths), done=set(),
                     thumbs=[p.get("thumb") for p in pages])

    def page_done(index: int) -> None:
        _progress_page_done(token, index)

    rows = None
    fallback_reason = ""
    if claude_reader.is_enabled():
        claude_engine = f"Claude vision ({claude_reader.model_name()})"
        try:
            if detect_tables:
                tables = claude_reader.read_tables_from_pages(page_paths, on_page_done=page_done)
                if not tables:
                    tables = [{"title": "", "columns": ["Column 1", "Column 2", "Column 3"],
                               "rows": [["", "", ""]], "page": 1}]
                for table in tables:
                    table["page"] = table.get("page") or 1
                    table["image"] = pages[table["page"] - 1].get("view")
                return {**files, "tables": tables, "used_engine": claude_engine, "fallback_reason": "",
                        "pages_note": _pages_note(len(page_paths), total_pages, is_pdf)}, None
            rows = claude_reader.read_rows_from_pages(page_paths, on_page_done=page_done)
            provenance = [{} for _ in rows]
            used_engine = claude_engine
            pages_read = len(page_paths)
        except claude_reader.ClaudeReadError as exc:
            app.logger.warning("Claude read failed, falling back to OCR fusion: %s", exc)
            fallback_reason = str(exc)
    else:
        fallback_reason = "no API key is set"

    if rows is None:
        # Tesseract-on-cleaned (row structure) + EasyOCR-on-raw (clean values),
        # each engine called once per page, then fused with provenance -- which
        # engine won each field -- so the review UI can show what fusion
        # auto-corrected.
        rows, provenance = [], []
        fallback_pages = page_paths[:MAX_FALLBACK_PDF_PAGES]
        _progress_update(token, total=len(fallback_pages), done=set(), fallback=True)
        for i, page_path in enumerate(fallback_pages):
            clean_path = first_clean if i == 0 else clean_page(page_path)
            if clean_path is not None:
                tesseract_text, easyocr_text = run_engines(page_path, processed_path=clean_path)
                page_rows, page_prov = fuse_from_texts(tesseract_text, easyocr_text, return_provenance=True)
                rows.extend(page_rows)
                provenance.extend(page_prov)
            page_done(i)
        used_engine = "fused (tesseract + easyocr)"
        if fallback_reason and claude_reader.is_enabled():
            used_engine += f" -- Claude unavailable: {fallback_reason}"
        pages_read = len(fallback_pages)

    if not rows:
        # Never dead-end the owner with nothing to correct -- give one blank row.
        rows = [{k: "" for k in FIELDNAMES}]
        provenance = [{}]
        used_engine = "none (manual entry)"

    return {**files, "rows": rows, "provenance": provenance, "used_engine": used_engine,
            "fallback_reason": fallback_reason,
            "pages_note": _pages_note(pages_read, total_pages, is_pdf)}, None


def _save_rows(source_file: str, rows: list[dict]) -> int:
    """Insert non-blank *rows* into SQLite under *source_file*; return count saved."""
    created_at = datetime.now(timezone.utc).isoformat()
    db = get_db()
    saved = 0
    for row in rows:
        clean = {field: str(row.get(field, "") or "").strip() for field in FIELDNAMES}
        if not any(clean.values()):
            continue  # skip fully-blank rows (e.g. an unused manual-entry row)
        db.execute(
            """
            INSERT INTO records (date, item, qty, price, total, source_file, created_at)
            VALUES (:date, :item, :qty, :price, :total, :source_file, :created_at)
            """,
            {**clean, "source_file": source_file, "created_at": created_at},
        )
        saved += 1
    db.commit()
    return saved


@app.errorhandler(RequestEntityTooLarge)
def too_large(_exc):
    limit_mb = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
    message = tr("That file is too large. The limit is {mb} MB.", mb=limit_mb)
    if request.path.startswith("/api/") or request.path == "/upload":
        return jsonify({"error": message}), 413
    flash(message)
    return redirect(url_for("index"))


# ---------------------------------------------------------------------------
# Saved documents (the scans table)
# ---------------------------------------------------------------------------


def _scan_dict(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "name": row["name"],
        "source_file": row["source_file"],
        "engine": row["engine"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "tables": json.loads(row["tables_json"]),
    }


_SCAN_COLUMNS = "id, name, source_file, engine, tables_json, created_at, updated_at"


def _load_scan(scan_id: int) -> dict | None:
    row = get_db().execute(f"SELECT {_SCAN_COLUMNS} FROM scans WHERE id = ?", (scan_id,)).fetchone()
    return _scan_dict(row) if row else None


def _nice_date(iso: str | None) -> str:
    """'2 Oct 2026' from a stored ISO timestamp (or today)."""
    try:
        when = datetime.fromisoformat(iso) if iso else datetime.now(timezone.utc)
    except ValueError:
        when = datetime.now(timezone.utc)
    return f"{when.day} {when:%b %Y}"


def _scan_summary(scan: dict) -> dict:
    pages = sorted({t.get("page") or 1 for t in scan["tables"]})
    return {
        "id": scan["id"],
        "label": document_label(scan),
        "date": _nice_date(scan["updated_at"] or scan["created_at"]),
        "pages": len(pages),
        "rows": sum(len(t["rows"]) for t in scan["tables"]),
    }


def _known_names() -> list[str]:
    """Document names used before, newest first (suggested when naming a new one)."""
    rows = get_db().execute(
        "SELECT name, MAX(id) AS newest FROM scans WHERE name IS NOT NULL AND name != '' "
        "GROUP BY name ORDER BY newest DESC LIMIT 30"
    )
    return [row["name"] for row in rows]


# ---------------------------------------------------------------------------
# HTML routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    """Upload form, plus the most recently saved documents to reopen."""
    recent = [
        _scan_summary(_scan_dict(row))
        for row in get_db().execute(
            f"SELECT {_SCAN_COLUMNS} FROM scans ORDER BY COALESCE(updated_at, created_at) DESC, id DESC LIMIT 5"
        )
    ]
    return render_template("index.html", recent=recent, names=_known_names(), max_pdf_pages=MAX_PDF_PAGES)


@app.route("/lang/<code>")
def set_language(code: str):
    """Switch the interface language, then go back to the page it was chosen on."""
    target = request.args.get("next", "")
    if not target.startswith("/") or target.startswith("//") or "\\" in target:
        target = url_for("index")
    response = redirect(target)
    if code in LANGUAGES:
        response.set_cookie("lang", code, max_age=365 * 24 * 3600, samesite="Lax")
    return response


def _image_url(name: str | None) -> str | None:
    if name and secure_filename(name) == name and (PROCESSED_DIR / name).is_file():
        return url_for("processed_file", filename=name)
    return None


def _render_review(artifacts: dict, scan: dict | None = None):
    """The editable review page, for a fresh read or a saved document (*scan*)."""
    tables = artifacts.get("tables")
    page_urls: dict[int, str] = {}
    for page in artifacts.get("pages") or []:
        url = _image_url(page.get("view"))
        if url:
            page_urls[page["page"]] = url
    for table in tables or []:
        url = _image_url(table.get("image"))
        if url:
            page_urls.setdefault(table.get("page") or 1, url)
    if 1 not in page_urls and artifacts.get("raw_filename"):
        page_urls[1] = url_for("uploaded_file", filename=artifacts["raw_filename"])

    groups = []
    if tables:
        by_page: dict[int, list] = {}
        for t, table in enumerate(tables):
            by_page.setdefault(table.get("page") or 1, []).append((t, table))
        groups = [{"page": page, "image": page_urls.get(page), "tables": by_page[page]} for page in sorted(by_page)]
        cells = [cell for table in tables for row in table["rows"] for cell in row]
        n_rows = sum(len(table["rows"]) for table in tables)
    else:
        cells = [str(row.get(field, "")) for row in artifacts["rows"] for field in FIELDNAMES]
        n_rows = len(artifacts["rows"])

    append_to = None
    if artifacts.get("append_to"):
        target = _load_scan(artifacts["append_to"])
        if target:
            append_to = {"id": target["id"], "label": document_label(target)}

    used_engine = artifacts["used_engine"]
    return render_template(
        "process.html",
        tables=tables,
        groups=groups,
        rows=artifacts.get("rows"),
        page_images=sorted(page_urls.items()),
        source_file=artifacts["source_file"],
        used_engine=used_engine,
        free_reader=used_engine.startswith(("fused", "none")),
        fallback_reason=artifacts.get("fallback_reason", ""),
        processed_url=(url_for("processed_file", filename=artifacts["processed_filename"])
                       if artifacts.get("processed_filename") else None),
        pages_note=artifacts.get("pages_note", ""),
        doc_name=(scan["name"] if scan else artifacts.get("doc_name")) or "",
        scan=scan,
        append_to=append_to,
        names=_known_names(),
        summary={"pages": max(len(groups), len(page_urls), 1), "rows": n_rows,
                 "unsure": sum("?" in cell for cell in cells)},
    )


def _review_path(token: str) -> Path | None:
    """Where the read result for upload *token* is kept, or None if invalid."""
    name = secure_filename(token or "")
    if not name or name != token:
        return None
    return PROCESSED_DIR / f"{name}.review.json"


def _document_options() -> dict:
    """The optional name and target document sent along with an upload."""
    try:
        append_to = int(request.form.get("append_to") or 0)
    except ValueError:
        append_to = 0
    return {"doc_name": request.form.get("name", "").strip()[:120], "append_to": append_to or None}


@app.route("/process", methods=["POST"])
def process():
    """Run the pipeline and show the editable table(s).

    Plain form submit (no JavaScript): takes the file and renders the review.

    With a ``token`` from /upload (the page's script uploads first so it can
    show real upload progress): reads that file, keeps the result, and
    answers ``{"review_url": ...}`` -- or ``{"error": ...}`` -- so the script
    can open the review page as a normal page load. While it reads, the
    script polls /progress/<token> to show each page finishing.
    """
    token = request.form.get("token")
    if token:
        source_path = _uploaded_path(token)
        if source_path is None:
            return jsonify({"error": tr("That upload could not be found. Please choose the file again.")}), 404
        _progress_update(token, stage="preparing")
        try:
            artifacts, error = _run_pipeline_on_file(source_path, detect_tables=True, token=token)
        except Exception:
            _progress_update(token, stage="error", error=tr("The server hit an error while reading this file."))
            raise
        if error:
            _progress_update(token, stage="error", error=error)
            return jsonify({"error": error}), 400
        artifacts.update(_document_options())
        _review_path(token).write_text(json.dumps(artifacts, ensure_ascii=False), encoding="utf-8")
        review_url = url_for("review", token=token)
        _progress_update(token, stage="done", review_url=review_url)
        return jsonify({"review_url": review_url})

    artifacts, error = _run_pipeline_on_upload(request.files.get("image"), detect_tables=True)
    if error:
        flash(error)
        return redirect(url_for("index"))
    artifacts.update(_document_options())
    return _render_review(artifacts)


@app.route("/progress/<token>")
def progress(token: str):
    """How far the read of upload *token* has got: stage, pages done, thumbnails."""
    with _progress_lock:
        entry = _progress.get(token)
        entry = {**entry, "done": set(entry["done"])} if entry else None
    if entry is None:
        path = _review_path(token)
        if path is not None and path.is_file():
            return jsonify({"stage": "done", "review_url": url_for("review", token=token)})
        return jsonify({"stage": "unknown"}), 404
    return jsonify({
        "stage": entry["stage"],
        "total": entry["total"],
        "done": len(entry["done"]),
        "fallback": entry.get("fallback", False),
        "pages": [
            {"thumb": url_for("processed_file", filename=thumb) if thumb else None, "done": i in entry["done"]}
            for i, thumb in enumerate(entry["thumbs"])
        ],
        "review_url": entry.get("review_url"),
        "error": entry.get("error"),
    })


@app.route("/review/<token>")
def review(token: str):
    """Show a result read earlier via /process (refresh-safe)."""
    path = _review_path(token)
    if path is None or not path.is_file():
        flash(tr("That review could not be found. Please upload the file again."))
        return redirect(url_for("index"))
    return _render_review(json.loads(path.read_text(encoding="utf-8")))


@app.route("/upload", methods=["POST"])
def upload():
    """Save an upload without reading it; returns ``{"token": ...}`` for /process."""
    source_path, error = _save_upload(request.files.get("image"))
    if error:
        return jsonify({"error": error}), 400
    return jsonify({"token": source_path.name})


@app.route("/save", methods=["POST"])
def save():
    """Persist the (possibly corrected) rows from /process into SQLite."""
    source_file = request.form.get("source_file", "").strip()
    try:
        num_rows = int(request.form.get("num_rows", "0"))
    except ValueError:
        num_rows = 0

    rows = [
        {field: request.form.get(f"{field}_{i}", "") for field in FIELDNAMES}
        for i in range(min(max(num_rows, 0), _MAX_ROWS))
    ]
    saved = _save_rows(source_file, rows)

    flash(tr("Saved {n} record(s) from {source}.", n=saved, source=source_file or tr("upload")))
    return redirect(url_for("index"))


# Bounds on the dynamic review form, so a crafted POST can't make the server
# loop over millions of fields.
_MAX_TABLES, _MAX_COLS, _MAX_ROWS, _MAX_PAGE = 200, 50, 2000, 10000
_MAX_ID = 2**62


def _form_int(name: str, upper: int) -> int:
    try:
        return max(0, min(int(request.form.get(name, "0")), upper))
    except ValueError:
        return 0


@app.route("/save-tables", methods=["POST"])
def save_tables():
    """Persist tables read with the page's own headings into the scans table.

    Saves a new document, or -- with ``scan_id`` -- the edits to a saved one,
    or -- with ``append_to`` -- adds these pages after a saved one's pages.
    """
    source_file = request.form.get("source_file", "").strip()
    engine = request.form.get("engine", "").strip()
    name = request.form.get("name", "").strip()[:120] or None
    editing = _load_scan(_form_int("scan_id", _MAX_ID)) if request.form.get("scan_id") else None
    adding_to = _load_scan(_form_int("append_to", _MAX_ID)) if request.form.get("append_to") else None

    tables = []
    for t in range(_form_int("num_tables", _MAX_TABLES)):
        n_cols = _form_int(f"num_cols_{t}", _MAX_COLS)
        n_rows = _form_int(f"num_rows_{t}", _MAX_ROWS)
        columns = [request.form.get(f"col_{t}_{c}", "").strip() for c in range(n_cols)]
        rows = []
        for r in range(n_rows):
            row = [request.form.get(f"cell_{t}_{r}_{c}", "").strip() for c in range(n_cols)]
            if any(row):
                rows.append(row)
        if rows:
            image = request.form.get(f"image_{t}", "")
            tables.append({
                "title": request.form.get(f"title_{t}", "").strip(),
                "columns": [col or f"Column {i + 1}" for i, col in enumerate(columns)],
                "rows": rows,
                "page": _form_int(f"page_{t}", _MAX_PAGE) or 1,
                "image": image if _image_url(image) else None,
            })

    target = editing or adding_to
    if not tables:
        flash(tr("Nothing to save: every row was empty."))
        return redirect(url_for("document", scan_id=target["id"]) if target else url_for("index"))

    db = get_db()
    now = datetime.now(timezone.utc).isoformat()
    n_rows = sum(len(t["rows"]) for t in tables)
    if editing:
        db.execute("UPDATE scans SET name = ?, tables_json = ?, updated_at = ? WHERE id = ?",
                   (name, json.dumps(tables, ensure_ascii=False), now, editing["id"]))
        scan_id = editing["id"]
        message = tr("Saved your changes: {rows} row(s) in {tables} table(s).", rows=n_rows, tables=len(tables))
    elif adding_to:
        offset = max((t.get("page") or 1 for t in adding_to["tables"]), default=0)
        for table in tables:
            table["page"] += offset
        new_pages = len({t["page"] for t in tables})
        db.execute("UPDATE scans SET name = ?, tables_json = ?, updated_at = ? WHERE id = ?",
                   (name or adding_to["name"], json.dumps(adding_to["tables"] + tables, ensure_ascii=False),
                    now, adding_to["id"]))
        scan_id = adding_to["id"]
        message = tr("Added {pages} page(s) to {name}.", pages=new_pages,
                     name=name or document_label(adding_to))
    else:
        cursor = db.execute(
            "INSERT INTO scans (name, source_file, engine, tables_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (name, source_file, engine, json.dumps(tables, ensure_ascii=False), now),
        )
        scan_id = cursor.lastrowid
        message = tr("Saved {rows} row(s) in {tables} table(s) from {source}.",
                     rows=n_rows, tables=len(tables), source=name or source_file or tr("upload"))
    db.commit()

    flash(message)
    return redirect(url_for("document", scan_id=scan_id, saved=1))


@app.route("/documents/<int:scan_id>")
def document(scan_id: int):
    """A saved document: download, share, copy, edit, or add pages to it."""
    scan = _load_scan(scan_id)
    if scan is None:
        flash(tr("That document could not be found."))
        return redirect(url_for("index"))

    pages: dict[int, dict] = {}
    for table in scan["tables"]:
        page = pages.setdefault(table.get("page") or 1, {"tables": [], "image": None, "thumb": None})
        page["tables"].append({"title": table["title"], "rows": len(table["rows"]), "cols": len(table["columns"])})
        if table.get("image") and not page["image"]:
            page["image"] = _image_url(table["image"])
            page["thumb"] = _image_url(_thumb_for(table["image"]))

    summary = _scan_summary(scan)
    return render_template(
        "document.html",
        scan=scan,
        summary=summary,
        pages=sorted(pages.items()),
        saved=request.args.get("saved") == "1",
        download_stem=_download_stem(summary["label"], scan["created_at"]),
        names=_known_names(),
        max_pdf_pages=MAX_PDF_PAGES,
    )


@app.route("/documents/<int:scan_id>/edit")
def edit_document(scan_id: int):
    """Reopen a saved document in the review page to fix something."""
    scan = _load_scan(scan_id)
    if scan is None:
        flash(tr("That document could not be found."))
        return redirect(url_for("index"))
    artifacts = {"tables": scan["tables"], "source_file": scan["source_file"] or "",
                 "used_engine": scan["engine"] or ""}
    return _render_review(artifacts, scan=scan)


@app.route("/documents/<int:scan_id>/export/<fmt>")
def export_document(scan_id: int, fmt: str):
    """Download one saved document. ``?tidy=0`` keeps cells exactly as written."""
    scan = _load_scan(scan_id)
    if scan is None or fmt not in ("xlsx", "csv", "json"):
        flash(tr("That document could not be found."))
        return redirect(url_for("index"))
    stem = _download_stem(document_label(scan), scan["created_at"])
    return _download(fmt, [], _RECORD_COLUMNS, [scan], stem, tidy=request.args.get("tidy") != "0")


# ---------------------------------------------------------------------------
# JSON API -- used by the companion mobile app (app/). Same pipeline and
# SQLite store as the HTML routes above, just JSON in/out instead of
# server-rendered forms.
# ---------------------------------------------------------------------------


@app.route("/api/health")
def api_health():
    """Liveness probe -- the mobile app pings this to detect a cold Space waking up."""
    return jsonify({"status": "ok"})


@app.route("/api/scan", methods=["POST"])
def api_scan():
    """Upload a ledger photo, run the pipeline, return rows + provenance as JSON."""
    artifacts, error = _run_pipeline_on_upload(request.files.get("image"))
    if error:
        return jsonify({"error": error}), 400

    return jsonify({
        "rows": artifacts["rows"],
        "provenance": artifacts["provenance"],
        "used_engine": artifacts["used_engine"],
        "pages_note": artifacts["pages_note"],
        "source_file": artifacts["source_file"],
        "raw_url": url_for("uploaded_file", filename=artifacts["raw_filename"]),
        "processed_url": url_for("processed_file", filename=artifacts["processed_filename"]),
    })


@app.route("/api/save", methods=["POST"])
def api_save():
    """Persist (possibly corrected) rows into SQLite. Body: {source_file, rows}."""
    body = request.get_json(silent=True) or {}
    source_file = str(body.get("source_file", "")).strip()
    rows = body.get("rows", [])
    if not isinstance(rows, list):
        return jsonify({"error": "'rows' must be a list."}), 400

    saved = _save_rows(source_file, rows)
    return jsonify({"saved": saved})


_RECORD_COLUMNS = ["id", "date", "item", "qty", "price", "total", "source_file", "created_at"]
_FILENAME_BAD_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
_MIMETYPES = {
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "csv": "text/csv",
    "json": "application/json",
}


def _download_stem(label: str, iso_date: str | None = None) -> str:
    """'Seed book – 2 Oct 2026': the document's name and date, safe as a file name."""
    label = _FILENAME_BAD_CHARS.sub("-", label).strip(" .") or "MSME Ledger"
    return f"{label[:80]} – {_nice_date(iso_date)}"


def _csv_text(records: list[dict], columns: list[str], scans: list[dict]) -> str:
    """Records first, then each scanned table as its own block: a label line
    (document, title and page), its header row, its rows."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    if records or not scans:
        writer.writerow(columns)
        writer.writerows([r[c] for c in columns] for r in records)
    for scan in scans:
        label = document_label(scan)
        for table in scan["tables"]:
            if buf.tell():
                writer.writerow([])
            parts = [label] + ([table["title"]] if table["title"] else [])
            writer.writerow([" - ".join(parts) + f" (page {table.get('page') or 1})"])
            writer.writerow(table["columns"])
            writer.writerows(table["rows"])
    return buf.getvalue()


def _download(fmt: str, records: list[dict], columns: list[str], scans: list[dict], stem: str, tidy: bool = True):
    if fmt == "xlsx":
        data = build_workbook(records, columns, scans, tidy=tidy)
    elif fmt == "csv":
        # With a byte-order mark, Excel opens the CSV as UTF-8.
        data = _csv_text(records, columns, scans).encode("utf-8-sig")
    else:
        data = json.dumps({"records": records, "scans": scans}, indent=2, ensure_ascii=False).encode("utf-8")
    return send_file(io.BytesIO(data), mimetype=_MIMETYPES[fmt], as_attachment=True,
                     download_name=f"{stem}.{fmt}")


@app.route("/export/<fmt>")
def export(fmt: str):
    """Download everything saved as Excel, CSV or JSON.

    Covers both the fixed-field records and the scans saved with their own
    headings. Excel puts each page of each scan on its own sheet, with an
    "All pages" sheet per multi-page document (see web/exports.py). JSON is
    {"records": [...], "scans": [...]}. A single document can be downloaded
    from its own page instead (/documents/<id>/export/<fmt>).
    """
    if fmt not in _MIMETYPES:
        flash(tr("Unknown export format '{fmt}'. Use 'xlsx', 'csv' or 'json'.", fmt=fmt))
        return redirect(url_for("index"))

    db = get_db()
    records = [dict(row) for row in db.execute(f"SELECT {', '.join(_RECORD_COLUMNS)} FROM records ORDER BY id")]
    scans = [_scan_dict(row) for row in db.execute(f"SELECT {_SCAN_COLUMNS} FROM scans ORDER BY id")]

    stem = _download_stem(document_label(scans[0])) if len(scans) == 1 and not records else _download_stem("MSME Ledger")
    n_rows = len(records) + sum(len(t["rows"]) for s in scans for t in s["tables"])
    flash(tr("Exported {n} row(s) as {file}.", n=n_rows, file=f"{stem}.{fmt}"))
    return _download(fmt, records, _RECORD_COLUMNS, scans, stem, tidy=request.args.get("tidy") != "0")


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
