"""
extraction/fields.py -- turn raw OCR text for one ledger into structured rows.

Each row is a dict with keys: date, item, qty, price, total.

Two OCR output shapes are handled automatically:

* **Flat block** (Tesseract): each ledger row sits on its own line, e.g.
  ``2026-01-03 Maize meal 2 3.50 7.00``.  Parsed line by line; lines without a
  date (title, header, footer) are skipped.
* **Token-per-line** (EasyOCR with ``paragraph=False``): each detection box is
  its own line, so a single row is spread across many lines.  All tokens are
  flattened and regrouped into rows on each date boundary.

Regex does the heavy lifting for the typed fields (date, decimals, integers).
spaCy is used only as a fallback for the item name when no leftover text tokens
remain; it degrades gracefully to a no-op if spaCy or its model is not installed.

Usage (library):
    from extraction.fields import extract_fields, rows_to_csv, rows_to_json
    rows = extract_fields(ocr_text)

Usage (batch):
    python -m extraction.fields
    python -m extraction.fields --ocr-raw results/ocr_raw \\
        --samples data/samples --out results/extracted
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

FIELDNAMES = ["date", "item", "qty", "price", "total"]
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

# ---------------------------------------------------------------------------
# Typed-field patterns
# ---------------------------------------------------------------------------

# YYYY-MM-DD and common variants (also DD-MM-YYYY), separators -, /, or .
_DATE_RE = re.compile(
    r"\b(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}"
    r"|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})\b"
)
# A decimal money amount, optional leading currency symbol, . or , separator.
_MONEY_RE = re.compile(r"^[£$€]?\d+[.,]\d{1,2}$")
# A standalone integer (quantity). "2kg" is rejected -> stays part of the item.
_INT_RE = re.compile(r"^\d+$")


def _find_date(text: str) -> str:
    """Return the first date substring in *text*, or ''."""
    m = _DATE_RE.search(text)
    return m.group(0) if m else ""


def _is_date_token(token: str) -> bool:
    """True if the whole token is a date."""
    return bool(_DATE_RE.fullmatch(token))


def _is_money_token(token: str) -> bool:
    return bool(_MONEY_RE.match(token))


def _norm_money(token: str) -> str:
    """Strip a currency symbol and normalise the decimal comma to a dot."""
    m = re.search(r"\d+[.,]\d{1,2}", token)
    return m.group(0).replace(",", ".") if m else ""


def _line_has_money(line: str) -> bool:
    return any(_is_money_token(t) for t in line.split())


# ---------------------------------------------------------------------------
# spaCy fallback for the item name (optional dependency)
# ---------------------------------------------------------------------------

_NLP = None
_SPACY_TRIED = False


def _get_nlp():
    """Load ``en_core_web_sm`` once; return None if spaCy/model unavailable."""
    global _NLP, _SPACY_TRIED
    if _SPACY_TRIED:
        return _NLP
    _SPACY_TRIED = True
    try:
        import spacy  # noqa: WPS433 (optional import)
        try:
            _NLP = spacy.load("en_core_web_sm")
        except OSError:
            _NLP = None  # model not downloaded
    except ImportError:
        _NLP = None  # spaCy not installed
    return _NLP


def _spacy_item_fallback(text: str) -> str:
    """Best-effort item name from *text* using spaCy noun chunks / entities.

    Only invoked when regex leaves no leftover text tokens for the item. Returns
    '' if spaCy is unavailable or nothing suitable is found.
    """
    nlp = _get_nlp()
    if nlp is None:
        return ""
    doc = nlp(text)
    # Prefer PRODUCT / ORG / WORK_OF_ART entities, else the first noun chunk.
    for ent in doc.ents:
        if ent.label_ in {"PRODUCT", "ORG", "WORK_OF_ART"}:
            return ent.text.strip()
    for chunk in doc.noun_chunks:
        cleaned = " ".join(t.text for t in chunk if t.is_alpha).strip()
        if cleaned:
            return cleaned
    return ""


# ---------------------------------------------------------------------------
# Core row classifier (shared by both parse shapes)
# ---------------------------------------------------------------------------


def _classify_row_tokens(tokens: list[str]) -> dict | None:
    """Classify one row's worth of tokens into a field dict.

    A row is only emitted if it contains a date (the reliable row anchor).
    Numbers are assigned end-anchored: the last decimal is the total, the one
    before it the price -- this survives a dropped leading amount better than a
    front-anchored guess. Missing fields become '' rather than raising.
    """
    joined = " ".join(tokens)
    date = _find_date(joined)
    if not date:
        return None

    monies: list[str] = []
    qty = ""
    item_words: list[str] = []

    for tok in tokens:
        if _is_date_token(tok):
            continue  # already captured as the anchor
        if _is_money_token(tok):
            monies.append(_norm_money(tok))
        elif _INT_RE.match(tok) and not qty:
            qty = tok  # first standalone integer is the quantity
        elif _INT_RE.match(tok):
            item_words.append(tok)  # extra bare integers fall back to item text
        else:
            item_words.append(tok)

    total = monies[-1] if monies else ""
    price = monies[-2] if len(monies) >= 2 else ""

    item = " ".join(item_words).strip()
    if not item:
        # Regex left nothing for the item -> try spaCy on the full row text.
        item = _spacy_item_fallback(joined)

    return {"date": date, "item": item, "qty": qty, "price": price, "total": total}


# ---------------------------------------------------------------------------
# Shape detection + the two parsers
# ---------------------------------------------------------------------------


def _detect_shape(lines: list[str]) -> str:
    """Return 'flat' if whole rows sit on single lines, else 'tokens'.

    Signal for 'flat': at least two lines that each contain a date, a decimal
    amount, and three or more whitespace tokens -- i.e. a complete row per line.
    """
    complete = sum(
        1 for ln in lines
        if _find_date(ln) and _line_has_money(ln) and len(ln.split()) >= 3
    )
    return "flat" if complete >= 2 else "tokens"


def _parse_flat(lines: list[str]) -> list[dict]:
    """Parse Tesseract-style output: one ledger row per line."""
    rows = []
    for ln in lines:
        if not _find_date(ln):
            continue  # skip title / header / footer lines
        row = _classify_row_tokens(ln.split())
        if row is not None:
            rows.append(row)
    return rows


def _parse_tokens(lines: list[str]) -> list[dict]:
    """Parse EasyOCR-style output: flatten all tokens, group on date boundaries."""
    tokens: list[str] = []
    for ln in lines:
        tokens.extend(ln.split())

    rows = []
    current: list[str] | None = None
    for tok in tokens:
        if _is_date_token(tok):
            if current is not None:
                row = _classify_row_tokens(current)
                if row is not None:
                    rows.append(row)
            current = [tok]  # start a new row at each date
        elif current is not None:
            current.append(tok)
        # tokens before the first date (title/header) are ignored
    if current is not None:
        row = _classify_row_tokens(current)
        if row is not None:
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_fields(ocr_text: str) -> list[dict]:
    """Extract structured ledger rows from raw OCR *ocr_text*.

    Returns a list of dicts with keys ``date, item, qty, price, total``.
    Never raises on malformed input -- an empty or unparseable string yields [].
    """
    if not ocr_text or not ocr_text.strip():
        return []
    lines = [ln.strip() for ln in ocr_text.splitlines() if ln.strip()]
    if not lines:
        return []
    if _detect_shape(lines) == "flat":
        return _parse_flat(lines)
    return _parse_tokens(lines)


def rows_to_csv(rows: list[dict], out_path: str | os.PathLike) -> None:
    """Write *rows* to *out_path* as CSV with the canonical column order."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDNAMES})


def rows_to_json(rows: list[dict], out_path: str | os.PathLike) -> None:
    """Write *rows* to *out_path* as pretty-printed JSON."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    normalised = [{k: row.get(k, "") for k in FIELDNAMES} for row in rows]
    out.write_text(json.dumps(normalised, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Batch entry point
# ---------------------------------------------------------------------------


def _stems_from_ocr_raw(ocr_raw_dir: Path) -> list[tuple[str, str]]:
    """Return (stem, text) pairs from every .txt in *ocr_raw_dir*."""
    pairs = []
    for txt in sorted(ocr_raw_dir.glob("*.txt")):
        pairs.append((txt.stem, txt.read_text(encoding="utf-8")))
    return pairs


def _stems_from_live_ocr(samples_dir: Path) -> list[tuple[str, str]]:
    """Run OCR live over images in *samples_dir*, returning (stem, text) pairs.

    Uses EasyOCR on the raw sample images (the raw-image engine per the project's
    findings). Imports lazily so the extraction library has no hard OCR dependency.
    """
    try:
        from ocr.compare import run_easyocr
    except ImportError as exc:  # pragma: no cover - depends on optional deps
        print(f"Live OCR unavailable ({exc}). Add .txt files to results/ocr_raw/ instead.")
        return []

    images = sorted(
        p for p in samples_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    )
    pairs = []
    for img in images:
        print(f"  [ocr] {img.name}")
        pairs.append((img.stem, run_easyocr(img)))
    return pairs


def _run(ocr_raw_dir: Path, samples_dir: Path, out_dir: Path) -> None:
    """Extract fields for every available ledger and write CSV + JSON per stem."""
    if ocr_raw_dir.exists() and any(ocr_raw_dir.glob("*.txt")):
        print(f"Reading OCR text from {ocr_raw_dir}")
        pairs = _stems_from_ocr_raw(ocr_raw_dir)
    elif samples_dir.exists():
        print(f"No .txt in {ocr_raw_dir}; running OCR live on {samples_dir}")
        pairs = _stems_from_live_ocr(samples_dir)
    else:
        print(f"Nothing to do: neither {ocr_raw_dir} nor {samples_dir} has input.")
        return

    if not pairs:
        print("No ledgers found to extract.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    for stem, text in pairs:
        rows = extract_fields(text)
        rows_to_csv(rows, out_dir / f"{stem}.csv")
        rows_to_json(rows, out_dir / f"{stem}.json")
        print(f"  [ok] {stem}: {len(rows)} row(s) -> {out_dir/f'{stem}.csv'}")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract structured ledger rows from raw OCR text.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Reads results/ocr_raw/<stem>.txt when present, otherwise runs OCR\n"
            "live over data/samples/. Writes results/extracted/<stem>.{csv,json}."
        ),
    )
    parser.add_argument("--ocr-raw", default="results/ocr_raw",
                        help="Folder of OCR .txt files (default: results/ocr_raw)")
    parser.add_argument("--samples", default="data/samples",
                        help="Raw images for live OCR fallback (default: data/samples)")
    parser.add_argument("--out", default="results/extracted",
                        help="Output folder for CSV/JSON (default: results/extracted)")
    args = parser.parse_args()

    _run(Path(args.ocr_raw), Path(args.samples), Path(args.out))
