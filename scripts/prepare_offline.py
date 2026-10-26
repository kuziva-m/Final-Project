"""
scripts/prepare_offline.py -- get a computer ready to run the app with no internet.

Run once while the computer still has internet:

    python scripts/prepare_offline.py

It downloads EasyOCR's model files into models/easyocr (so the first scan
doesn't try to fetch them), and checks Tesseract, the spaCy model and the
bundled fonts. Run it again any time to re-check; it only downloads what is
missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

problems: list[str] = []


def report(ok: bool, what: str, fix: str = "") -> None:
    print(("  OK    " if ok else "  NEEDS ") + what)
    if not ok:
        problems.append(f"{what}: {fix}" if fix else what)


def main() -> int:
    print("Checking this computer for offline use...\n")

    report(sys.version_info >= (3, 10), f"Python 3.10 or newer (found {sys.version.split()[0]})",
           "install Python 3.11 from python.org")

    try:
        from ocr import compare
    except ImportError as exc:
        report(False, "Python packages", f"run: pip install -r requirements.txt ({exc})")
        return finish()

    import pytesseract

    try:
        version = pytesseract.get_tesseract_version()
        report(True, f"Tesseract {version}")
    except Exception:
        report(False, "Tesseract OCR program",
               "install it (Windows: the UB Mannheim installer), or set LEDGER_TESSERACT_CMD to tesseract.exe")

    missing = [name for name in compare.EASYOCR_MODEL_FILES if not (compare.EASYOCR_MODEL_DIR / name).is_file()]
    if missing:
        print(f"  ...   downloading EasyOCR models into {compare.EASYOCR_MODEL_DIR} (about 100 MB)")
    try:
        compare._get_easyocr_reader()
        report(all((compare.EASYOCR_MODEL_DIR / name).is_file() for name in compare.EASYOCR_MODEL_FILES),
               "EasyOCR model files", "connect to the internet and run this script again")
    except Exception as exc:
        report(False, "EasyOCR model files", f"connect to the internet and run this script again ({exc})")

    try:
        import spacy

        spacy.load("en_core_web_sm")
        report(True, "spaCy English model (optional)")
    except Exception:
        print("  --    spaCy English model not installed (optional; run: python -m spacy download en_core_web_sm)")

    fonts = list((ROOT / "web" / "static" / "fonts").glob("*.woff2"))
    report(len(fonts) >= 6, "Fonts bundled with the app", "get the web/static/fonts folder from GitHub again")

    return finish()


def finish() -> int:
    print()
    if problems:
        print("Not ready yet:")
        for problem in problems:
            print("  - " + problem)
        return 1
    print("Ready. Start the app with start_offline.bat (Windows) or: python -m web.app")
    return 0


if __name__ == "__main__":
    sys.exit(main())
