"""
ocr/offline.py -- the free reader's work on one page (Tesseract + EasyOCR, no internet).

For each page, in order:

1. If the page is ruled into columns, EasyOCR's words are placed into those
   cells (ocr/grid.py), so the table keeps the page's own columns and headings.
2. Otherwise Tesseract (on the cleaned page) and EasyOCR (on the photo) read
   it, and ocr/fusion.py turns the two into fixed Date / Item / Qty / Price /
   Total rows. (They run one after the other on purpose: run side by side,
   their thread pools fight over the processor and both slow to a crawl --
   74 s instead of 2 s in a test.)
3. If neither gives a row, the lines of text EasyOCR read are kept, so the
   person has something to correct instead of an empty table.

Big photos are shrunk first (LEDGER_OCR_MAX_EDGE, default 2000 pixels on the
longest side): a 12-megapixel phone photo is several times slower to clean
and read and no more legible.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ocr import grid as grid_reader
from ocr.compare import _get_easyocr_reader, run_tesseract
from ocr.fusion import fuse_from_texts


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


MAX_EDGE = max(800, _env_int("LEDGER_OCR_MAX_EDGE", 2000))

# One page at a time: two scans read side by side (two phones at once) fight
# over the processor and each takes far longer than waiting its turn.
_READ_LOCK = threading.Lock()


def shrink(image: np.ndarray, max_edge: int = MAX_EDGE) -> np.ndarray:
    """*image* scaled down so its longest side is at most *max_edge* pixels."""
    h, w = image.shape[:2]
    scale = max_edge / max(h, w)
    if scale >= 1.0:
        return image
    return cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)


def load_image(path: str | os.PathLike) -> np.ndarray | None:
    """The page photo at *path* as a colour image, shrunk for reading; None if unreadable."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return None if image is None else shrink(image)


@dataclass
class PageRead:
    table: dict | None = None  # {"columns", "rows", "unsure"} from the page's ruled columns
    rows: list[dict] = field(default_factory=list)        # fixed Date/Item/Qty/Price/Total rows
    provenance: list[dict] = field(default_factory=list)  # which reader won each field of each row
    lines: list[str] = field(default_factory=list)        # the text read, when nothing else came out


def read_page(page_path: Path, clean: Callable[[], Path | None], want_table: bool = True) -> PageRead:
    """Read one page with the free reader.

    *clean* returns the path of the page's cleaned image for Tesseract
    (making it if needed), or None if the page can't be read; it is only
    called when Tesseract is needed. With *want_table* False the ruled-table
    reader is skipped and only fixed rows are returned.
    """
    with _READ_LOCK:
        return _read_page(page_path, clean, want_table)


def _read_page(page_path: Path, clean: Callable[[], Path | None], want_table: bool) -> PageRead:
    image = load_image(page_path)
    if image is None:
        return PageRead()
    reader = _get_easyocr_reader()

    words = None
    tilt = 0.0
    grid = grid_reader.find_grid(image) if want_table else None
    if grid is not None:
        table, words = grid_reader.read_table(grid, image, reader)
        if table:
            return PageRead(table=table)
        tilt = grid.tilt()

    if words is None:
        words = grid_reader.read_words(reader, image)
    clean_path = clean()
    te_text = run_tesseract(clean_path) if clean_path is not None else ""

    eo_text = "\n".join(str(text) for _, text, _ in words)
    rows, provenance = fuse_from_texts(te_text, eo_text, return_provenance=True)
    if rows:
        return PageRead(rows=rows, provenance=provenance)
    return PageRead(lines=grid_reader.text_lines(words, tilt))


def warm_up() -> None:
    """Load EasyOCR's model now, so the first scan doesn't wait for it."""
    _get_easyocr_reader()
