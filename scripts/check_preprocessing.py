"""scripts/check_preprocessing.py -- what preprocessing does for Tesseract.

Reads the synthetic ledger three ways and scores each against its ground
truth: the raw photo, the cleaned image with the previous (buggy) deskew,
and the cleaned image with the fixed deskew.

    python scripts/check_preprocessing.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import preprocessing.clean as clean  # noqa: E402
from ocr.compare import cer, run_tesseract, wer  # noqa: E402
from scripts.check_deskew import previous_deskew  # noqa: E402

SRC = Path("data/samples/synthetic_ledger.png")
TRUTH = Path("data/ground_truth/synthetic_ledger.txt").read_text(encoding="utf-8").strip()
OUT = Path("data/debug")


def main() -> None:
    img = cv2.imread(str(SRC), cv2.IMREAD_UNCHANGED)
    fixed = OUT / "check_fixed_clean.png"
    clean.preprocess(img, save_path=fixed)
    old = OUT / "check_previous_clean.png"
    current = clean.deskew
    clean.deskew = previous_deskew  # same pipeline, only the old deskew swapped in
    try:
        clean.preprocess(img, save_path=old)
    finally:
        clean.deskew = current

    print(f"{'Tesseract reads':<36}{'CER':>8}{'WER':>8}   digits as read in the Bread row")
    results = []
    for key, name, path in (("raw", "raw photo (no preprocessing)", SRC),
                            ("previous", "cleaned, previous deskew", old),
                            ("fixed", "cleaned, fixed deskew", fixed)):
        text = run_tesseract(path)
        bread = next((l for l in text.splitlines() if "Bread" in l or "2026-01-12" in l), "(row not found)")
        results.append({"key": key, "name": name, "cer": cer(text, TRUTH), "wer": wer(text, TRUTH)})
        print(f"{name:<36}{cer(text, TRUTH):>8.1%}{wer(text, TRUTH):>8.1%}   {bread.strip()}")
    Path("results/preprocessing_check.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
