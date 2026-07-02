"""
ocr/compare.py -- OCR comparison and scoring module.

Pipeline rule (from empirical findings on phone-captured records):
    Tesseract  -> preprocessed image  (deskew + denoise + CLAHE + binarise)
    EasyOCR    -> raw image           (deep-learning model tolerates skew well)

Usage (batch):
    python -m ocr.compare
    python -m ocr.compare --samples data/samples --processed data/processed \\
                          --ground-truth data/ground_truth --results results

Usage (library):
    from ocr.compare import run_tesseract, run_easyocr, cer, wer, score_image
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import jiwer
import pytesseract
import easyocr
from PIL import Image

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# OCR engines (logic mirrored from notebooks/spike_ocr_test.ipynb)
# ---------------------------------------------------------------------------

_EASYOCR_READER = None


def _get_easyocr_reader() -> easyocr.Reader:
    """Build the EasyOCR reader once and cache it; construction downloads models."""
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        _EASYOCR_READER = easyocr.Reader(["en"], gpu=False)
    return _EASYOCR_READER


def run_tesseract(image_path: str | os.PathLike) -> str:
    """Return text recognised by Tesseract for the image at *image_path*."""
    return pytesseract.image_to_string(Image.open(image_path)).strip()


def run_easyocr(image_path: str | os.PathLike) -> str:
    """Return text recognised by EasyOCR for the image at *image_path*.

    Each detection box is returned as one line; paragraph merging is off so
    column structure is preserved for later analysis.
    """
    results = _get_easyocr_reader().readtext(str(image_path), detail=0, paragraph=False)
    return "\n".join(results).strip()


# ---------------------------------------------------------------------------
# Error-rate metrics
# ---------------------------------------------------------------------------


def cer(pred: str, truth: str) -> float:
    """Character Error Rate between *pred* and *truth*.

    Delegates to ``jiwer.cer(reference=truth, hypothesis=pred)``.
    Edge cases: both empty -> 0.0; truth empty, pred non-empty -> 1.0.
    Result is clamped to [0, 1] because insertions can push jiwer above 1.
    """
    if not truth and not pred:
        return 0.0
    if not truth:
        return 1.0
    return min(1.0, jiwer.cer(truth, pred))


def wer(pred: str, truth: str) -> float:
    """Word Error Rate between *pred* and *truth*.

    Delegates to ``jiwer.wer(reference=truth, hypothesis=pred)``.
    Edge cases: both empty -> 0.0; truth empty, pred non-empty -> 1.0.
    Result is clamped to [0, 1].
    """
    if not truth and not pred:
        return 0.0
    if not truth:
        return 1.0
    return min(1.0, jiwer.wer(truth, pred))


# ---------------------------------------------------------------------------
# Per-image scoring
# ---------------------------------------------------------------------------


def score_image(
    image_path: str | os.PathLike,
    truth_text: str | None,
) -> list[dict]:
    """Run both OCR engines on *image_path* and score against *truth_text*.

    Returns a list of two dicts -- one per engine -- each containing:

    =========  ============================================================
    filename   basename of *image_path*
    engine     ``'tesseract'`` or ``'easyocr'``
    cer        character error rate (``None`` when *truth_text* is ``None``)
    wer        word error rate      (``None`` when *truth_text* is ``None``)
    text       raw OCR output string
    =========  ============================================================

    .. note::
        This function runs both engines on the **same** path.  To apply the
        pipeline rule (Tesseract on preprocessed, EasyOCR on raw), call the
        engine functions directly -- see ``__main__`` for the canonical usage.
    """
    filename = Path(image_path).name
    rows: list[dict] = []
    for engine_name, run_fn in [("tesseract", run_tesseract), ("easyocr", run_easyocr)]:
        text = run_fn(image_path)
        if truth_text is not None:
            c = cer(text, truth_text)
            w = wer(text, truth_text)
        else:
            c = w = None
        rows.append({"filename": filename, "engine": engine_name,
                     "cer": c, "wer": w, "text": text})
    return rows


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_truth(ground_truth_dir: Path, stem: str) -> str | None:
    """Return stripped content of ``ground_truth_dir/<stem>.txt``, or None."""
    path = ground_truth_dir / f"{stem}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return None


def _collect_images(directory: Path) -> list[Path]:
    """Return sorted list of image files directly inside *directory*."""
    if not directory.exists():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    )


def _original_stem(processed_stem: str) -> str:
    """Strip the ``_clean`` suffix that ``preprocessing/clean.py`` appends."""
    if processed_stem.endswith("_clean"):
        return processed_stem[: -len("_clean")]
    return processed_stem


def _write_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Batch pipeline
# ---------------------------------------------------------------------------


def _run_pipeline(
    samples_dir: Path,
    processed_dir: Path,
    ground_truth_dir: Path,
    results_dir: Path,
) -> None:
    """Apply the pipeline rule over every image in *samples_dir* and write CSVs.

    Pipeline rule
    -------------
    Tesseract  runs on the **preprocessed** image (deskew corrects the baseline
               skew that causes Tesseract to drop decimal points and misread 5 as S).
    EasyOCR    runs on the **raw** image (its CNN features are rotation-invariant,
               so preprocessing adds no benefit and can introduce binarisation
               artefacts that hurt its confidence scores).

    Outputs
    -------
    results/ocr_comparison.csv  -- one row per image per engine
    results/summary.csv         -- mean CER / WER per engine
    Console                     -- summary table
    """
    raw_images = _collect_images(samples_dir)
    if not raw_images:
        print(f"No images found in {samples_dir}")
        return

    # Build stem -> processed path map.
    proc_by_stem: dict[str, Path] = {}
    for p in _collect_images(processed_dir):
        proc_by_stem[_original_stem(p.stem)] = p

    detail_rows: list[dict] = []

    for raw_path in raw_images:
        stem = raw_path.stem
        truth = _load_truth(ground_truth_dir, stem)
        if truth is None:
            print(f"  [no truth]     {raw_path.name} -- CER/WER skipped")

        proc_path = proc_by_stem.get(stem)
        if proc_path is None:
            print(f"  [no processed] {stem} -- Tesseract will run on raw")

        tess_source = proc_path if proc_path is not None else raw_path

        # Pipeline rule: separate engine/image pairings.
        tess_text = run_tesseract(tess_source)
        easy_text = run_easyocr(raw_path)

        for engine_name, text in [("tesseract", tess_text), ("easyocr", easy_text)]:
            detail_rows.append({
                "filename": raw_path.name,
                "engine":   engine_name,
                "cer":      cer(text, truth) if truth is not None else None,
                "wer":      wer(text, truth) if truth is not None else None,
            })

    detail_path = results_dir / "ocr_comparison.csv"
    _write_csv(detail_rows, detail_path, ["filename", "engine", "cer", "wer"])
    print(f"\nWrote {detail_path}")

    _write_summary(detail_rows, results_dir)


def _write_summary(detail_rows: list[dict], results_dir: Path) -> None:
    """Compute mean CER / WER per engine, write summary.csv, print to console."""
    acc: dict[str, dict] = defaultdict(lambda: {"cer_sum": 0.0, "wer_sum": 0.0, "n": 0})
    for row in detail_rows:
        if row["cer"] is None:
            continue
        e = row["engine"]
        acc[e]["cer_sum"] += row["cer"]
        acc[e]["wer_sum"] += row["wer"]
        acc[e]["n"] += 1

    summary_rows: list[dict] = []
    for engine in sorted(acc):
        n = acc[engine]["n"]
        summary_rows.append({
            "engine":   engine,
            "mean_cer": round(acc[engine]["cer_sum"] / n, 6) if n else None,
            "mean_wer": round(acc[engine]["wer_sum"] / n, 6) if n else None,
            "n_images": n,
        })

    summary_path = results_dir / "summary.csv"
    _write_csv(summary_rows, summary_path, ["engine", "mean_cer", "mean_wer", "n_images"])
    print(f"Wrote {summary_path}\n")

    col = 12
    print(f"{'Engine':<{col}}  {'Mean CER':>10}  {'Mean WER':>10}  {'N':>5}")
    print("-" * 44)
    for row in summary_rows:
        cer_s = f"{row['mean_cer']:.4f}" if row["mean_cer"] is not None else "N/A"
        wer_s = f"{row['mean_wer']:.4f}" if row["mean_wer"] is not None else "N/A"
        print(f"{row['engine']:<{col}}  {cer_s:>10}  {wer_s:>10}  {row['n_images']:>5}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score OCR engines against ground truth and write comparison CSVs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Drop images into --samples (and run preprocessing/clean.py first),\n"
            "add matching .txt files in --ground-truth, then re-run."
        ),
    )
    parser.add_argument("--samples",      default="data/samples",
                        help="Raw input images  (default: data/samples)")
    parser.add_argument("--processed",    default="data/processed",
                        help="Preprocessed images  (default: data/processed)")
    parser.add_argument("--ground-truth", default="data/ground_truth",
                        help="Ground-truth .txt files  (default: data/ground_truth)")
    parser.add_argument("--results",      default="results",
                        help="Output folder for CSVs  (default: results)")
    args = parser.parse_args()

    _run_pipeline(
        samples_dir=Path(args.samples),
        processed_dir=Path(args.processed),
        ground_truth_dir=Path(args.ground_truth),
        results_dir=Path(args.results),
    )
