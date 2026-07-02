"""
ocr/fusion.py -- rule-based fusion of two OCR passes into one clean ledger.

The two engines fail in complementary ways on phone-captured ledgers:

* **Tesseract on the preprocessed image** keeps the *table structure* -- one row
  per line, dates and item names in the right place -- but corrupts *digits*
  (drops decimal points, reads 5 as S, 7 as 1).
* **EasyOCR on the raw image** reads the *values* cleanly -- especially numbers --
  but its per-detection boxes *lose the table structure* (rows fragment, cells
  arrive out of order).

Fusion keeps each engine's strength:

1. Tesseract's rows are the **skeleton** (date, item, column layout per line).
2. EasyOCR supplies **clean values**, aligned back onto the skeleton by date /
   row order.
3. On a numeric field (qty, price, total) where the two disagree, EasyOCR's
   value wins *when it is a valid number of the right shape*; otherwise the
   Tesseract value is kept. Date and item always come from the skeleton.

Every decision is logged (which engine won each field) so the fusion is fully
transparent and auditable -- there is no model, only rules.

No new OCR logic lives here: OCR is delegated to ``ocr.compare.run_tesseract`` /
``run_easyocr`` and preprocessing to ``preprocessing.clean.preprocess``. Each
engine is called at most once per image.

Usage (library):
    from ocr.fusion import fuse
    rows = fuse("data/samples/ledger.jpg")          # -> [{date,item,qty,price,total}, ...]

Usage (scoring / batch):
    python -m ocr.fusion                            # score fusion vs each engine
"""

from __future__ import annotations

import argparse
import logging
import re
from collections import defaultdict
from pathlib import Path

from extraction.fields import (
    FIELDNAMES,
    _INT_RE,
    extract_fields,
    rows_to_csv,
)
from ocr.compare import (
    _collect_images,
    _write_csv,
    cer,
    run_easyocr,
    run_tesseract,
    wer,
)

logger = logging.getLogger("ocr.fusion")

_REPO_ROOT = Path(__file__).resolve().parent.parent

# A normalised money value produced by extraction (currency stripped, dot decimal).
_MONEY_VALUE_RE = re.compile(r"\d+\.\d{1,2}")


# ---------------------------------------------------------------------------
# Field-level decision helpers
# ---------------------------------------------------------------------------


def _norm_date(value: str) -> str:
    """Normalise a date for alignment: strip and unify separators to '-'."""
    return re.sub(r"[./]", "-", (value or "").strip())


def _valid_money(value: str) -> bool:
    return bool(value) and bool(_MONEY_VALUE_RE.fullmatch(value.strip()))


def _valid_int(value: str) -> bool:
    return bool(value) and bool(_INT_RE.fullmatch(value.strip()))


def _choose_numeric(te_val: str, eo_val: str, is_valid) -> tuple[str, str]:
    """Fuse one numeric field. Return ``(value, winner)``.

    Rule (transparent):
      * agree            -> keep the shared value, winner 'agree'
      * disagree + eo ok -> prefer EasyOCR (its documented strength is values)
      * disagree + eo bad-> keep Tesseract if valid, else fall back to eo/te
    """
    te_val = (te_val or "").strip()
    eo_val = (eo_val or "").strip()

    if te_val == eo_val:
        return te_val, "agree"

    if is_valid(eo_val):
        return eo_val, "easyocr"
    if is_valid(te_val):
        return te_val, "tesseract"

    # Neither is a clean number: keep whatever is non-empty, preferring EasyOCR.
    if eo_val:
        return eo_val, "easyocr"
    return te_val, "tesseract"


# ---------------------------------------------------------------------------
# Row alignment
# ---------------------------------------------------------------------------


def _align(te_rows: list[dict], eo_rows: list[dict]) -> list[tuple[dict, dict | None]]:
    """Pair each Tesseract skeleton row with an EasyOCR row.

    Primary key is the normalised date (robust to row reordering); rows the date
    match cannot place fall back to positional order among the leftovers. A
    skeleton row with no partner is paired with ``None`` (its values are kept).
    """
    # Index EasyOCR rows by date; each date's rows are consumed in order so
    # duplicate dates still pair one-to-one.
    eo_by_date: dict[str, list[dict]] = defaultdict(list)
    for row in eo_rows:
        eo_by_date[_norm_date(row.get("date", ""))].append(row)

    matched_ids: set[int] = set()
    matches: dict[int, dict] = {}
    for i, te in enumerate(te_rows):
        bucket = eo_by_date.get(_norm_date(te.get("date", "")))
        if bucket:
            eo = bucket.pop(0)
            matches[i] = eo
            matched_ids.add(id(eo))

    # Leftover EasyOCR rows, in original order, for positional fallback.
    leftovers = [r for r in eo_rows if id(r) not in matched_ids]

    aligned: list[tuple[dict, dict | None]] = []
    li = 0
    for i, te in enumerate(te_rows):
        if i in matches:
            aligned.append((te, matches[i]))
        elif li < len(leftovers):
            aligned.append((te, leftovers[li]))
            li += 1
        else:
            aligned.append((te, None))
    return aligned


# ---------------------------------------------------------------------------
# Text-level fusion (pure -- no OCR, fully testable)
# ---------------------------------------------------------------------------


def fuse_from_texts(
    tesseract_text: str,
    easyocr_text: str,
    return_provenance: bool = False,
):
    """Fuse raw OCR text from both engines into structured rows.

    Tesseract text is the row skeleton; EasyOCR text supplies clean values.
    Returns a list of ``{date, item, qty, price, total}`` dicts. When
    ``return_provenance`` is True, returns ``(rows, provenance)`` where each
    provenance entry records which engine won each numeric field.
    """
    te_rows = extract_fields(tesseract_text)  # skeleton: dates, items, layout
    eo_rows = extract_fields(easyocr_text)    # clean values, structure may be lost

    fused: list[dict] = []
    provenance: list[dict] = []

    for idx, (te, eo) in enumerate(_align(te_rows, eo_rows)):
        eo = eo or {}
        date = te.get("date", "")
        item = te.get("item", "")

        qty, qty_win = _choose_numeric(te.get("qty", ""), eo.get("qty", ""), _valid_int)
        price, price_win = _choose_numeric(te.get("price", ""), eo.get("price", ""), _valid_money)
        total, total_win = _choose_numeric(te.get("total", ""), eo.get("total", ""), _valid_money)

        logger.info(
            "row %d [%s %r]: qty<-%s price<-%s total<-%s",
            idx, date or "?", item or "?", qty_win, price_win, total_win,
        )

        fused.append({"date": date, "item": item, "qty": qty, "price": price, "total": total})
        provenance.append({"date": date, "qty": qty_win, "price": price_win, "total": total_win})

    if return_provenance:
        return fused, provenance
    return fused


# ---------------------------------------------------------------------------
# OCR wiring (one call per engine) + top-level fuse()
# ---------------------------------------------------------------------------


def run_engines(
    image_path,
    processed_path=None,
    processed_dir=None,
) -> tuple[str, str]:
    """Run each engine exactly once: Tesseract on cleaned, EasyOCR on raw.

    Returns ``(tesseract_text, easyocr_text)``. If no preprocessed image is
    supplied, the raw image is cleaned via ``preprocessing.clean.preprocess``
    (reused, not reimplemented) and cached under *processed_dir*.
    """
    import cv2  # local import: keeps text-level fusion importable without OpenCV
    from preprocessing.clean import preprocess

    image_path = Path(image_path)

    if processed_path is not None and Path(processed_path).exists():
        clean_path = Path(processed_path)
    else:
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        processed_dir = Path(processed_dir) if processed_dir else (_REPO_ROOT / "data" / "processed")
        clean_path = processed_dir / f"{image_path.stem}_clean.png"
        preprocess(img, save_path=clean_path)

    tesseract_text = run_tesseract(clean_path)  # structure-preserving pass
    easyocr_text = run_easyocr(image_path)      # value-preserving pass
    return tesseract_text, easyocr_text


def fuse(image_path, processed_path=None) -> list[dict]:
    """Fuse both OCR engines on *image_path* into clean ledger rows.

    Runs Tesseract on the preprocessed image and EasyOCR on the raw image (once
    each), then applies the rule-based fusion. Returns a list of
    ``{date, item, qty, price, total}`` dicts.
    """
    tesseract_text, easyocr_text = run_engines(image_path, processed_path)
    return fuse_from_texts(tesseract_text, easyocr_text)


# ---------------------------------------------------------------------------
# Scoring: fusion vs each engine, field-level CER/WER vs ground truth
# ---------------------------------------------------------------------------


def _rows_to_text(rows: list[dict]) -> str:
    """Flatten rows to a canonical text block for field-level CER/WER."""
    return "\n".join(
        " ".join((r.get(k, "") or "") for k in FIELDNAMES) for r in rows
    )


def _load_truth_fields(ground_truth_dir: Path, stem: str) -> list[dict] | None:
    """Read ``<stem>_fields.csv`` from *ground_truth_dir* as a list of row dicts."""
    import csv

    path = ground_truth_dir / f"{stem}_fields.csv"
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _cached_or_live_ocr(
    stem: str,
    raw_path: Path,
    processed_path: Path | None,
    cache_dir: Path,
) -> tuple[str, str]:
    """Return (tesseract_text, easyocr_text), reusing cached text when present.

    Cache files ``<stem>.tesseract.txt`` / ``<stem>.easyocr.txt`` avoid repeat
    OCR across runs. Missing cache -> run engines live (once each) and save.
    """
    tess_cache = cache_dir / f"{stem}.tesseract.txt"
    easy_cache = cache_dir / f"{stem}.easyocr.txt"

    if tess_cache.exists() and easy_cache.exists():
        logger.info("using cached OCR text for %s", stem)
        return (
            tess_cache.read_text(encoding="utf-8"),
            easy_cache.read_text(encoding="utf-8"),
        )

    tesseract_text, easyocr_text = run_engines(raw_path, processed_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tess_cache.write_text(tesseract_text, encoding="utf-8")
    easy_cache.write_text(easyocr_text, encoding="utf-8")
    return tesseract_text, easyocr_text


def score_fusion(
    samples_dir: Path,
    processed_dir: Path,
    ground_truth_dir: Path,
    results_dir: Path,
) -> list[dict]:
    """Score Tesseract-only, EasyOCR-only, and fused rows vs ground-truth fields.

    Writes ``results/fused/<stem>.csv`` (the fused ledger) and
    ``results/fusion_scores.csv`` (one row per engine per image), then prints a
    summary table so fusion can be compared against both engines at a glance.
    Returns the list of score rows.
    """
    images = _collect_images(samples_dir)
    if not images:
        print(f"No images in {samples_dir}; nothing to score.")
        return []

    fused_dir = results_dir / "fused"
    cache_dir = results_dir / "ocr_raw"
    scores: list[dict] = []

    for raw_path in images:
        stem = raw_path.stem
        truth_rows = _load_truth_fields(ground_truth_dir, stem)
        if truth_rows is None:
            print(f"  [no truth] {stem}: no {stem}_fields.csv -- skipped")
            continue

        processed_path = processed_dir / f"{stem}_clean.png"
        tesseract_text, easyocr_text = _cached_or_live_ocr(
            stem, raw_path, processed_path if processed_path.exists() else None, cache_dir
        )

        # Derive all three row sets from the SAME two OCR texts (no re-OCR).
        te_rows = extract_fields(tesseract_text)
        eo_rows = extract_fields(easyocr_text)
        fused_rows = fuse_from_texts(tesseract_text, easyocr_text)

        rows_to_csv(fused_rows, fused_dir / f"{stem}.csv")

        truth_text = _rows_to_text(truth_rows)
        for engine, rows in (
            ("tesseract", te_rows),
            ("easyocr", eo_rows),
            ("fused", fused_rows),
        ):
            pred_text = _rows_to_text(rows)
            scores.append({
                "stem": stem,
                "engine": engine,
                "cer": round(cer(pred_text, truth_text), 6),
                "wer": round(wer(pred_text, truth_text), 6),
            })

    _write_csv(scores, results_dir / "fusion_scores.csv", ["stem", "engine", "cer", "wer"])
    print(f"\nWrote {results_dir / 'fusion_scores.csv'}")
    print(f"Wrote fused ledgers to {fused_dir}/\n")
    _print_scores(scores)
    return scores


def _print_scores(scores: list[dict]) -> None:
    """Print mean CER/WER per engine and flag the best (lowest CER)."""
    acc: dict[str, dict] = defaultdict(lambda: {"cer": 0.0, "wer": 0.0, "n": 0})
    for row in scores:
        acc[row["engine"]]["cer"] += row["cer"]
        acc[row["engine"]]["wer"] += row["wer"]
        acc[row["engine"]]["n"] += 1

    means = {
        e: (a["cer"] / a["n"], a["wer"] / a["n"], a["n"])
        for e, a in acc.items() if a["n"]
    }
    if not means:
        return
    best = min(means, key=lambda e: means[e][0])

    print("Field-level accuracy vs ground truth (lower is better)")
    print(f"{'Engine':<12} {'Mean CER':>10} {'Mean WER':>10} {'N':>4}")
    print("-" * 40)
    # Show fixed engine order when present, then any extras.
    order = [e for e in ("tesseract", "easyocr", "fused") if e in means]
    order += [e for e in means if e not in order]
    for engine in order:
        mcer, mwer, n = means[engine]
        flag = "  <-- best" if engine == best else ""
        print(f"{engine:<12} {mcer:>10.4f} {mwer:>10.4f} {n:>4}{flag}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fuse Tesseract + EasyOCR and score fusion vs each engine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "For every image in --samples with a matching <stem>_fields.csv in\n"
            "--ground-truth, writes results/fused/<stem>.csv and appends to\n"
            "results/fusion_scores.csv, then prints the comparison table."
        ),
    )
    parser.add_argument("--samples", default="data/samples")
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--ground-truth", default="data/ground_truth")
    parser.add_argument("--results", default="results")
    parser.add_argument("--log-level", default="WARNING",
                        help="Logging level for per-field decisions (e.g. INFO)")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.WARNING),
                        format="%(name)s: %(message)s")

    score_fusion(
        samples_dir=Path(args.samples),
        processed_dir=Path(args.processed),
        ground_truth_dir=Path(args.ground_truth),
        results_dir=Path(args.results),
    )
