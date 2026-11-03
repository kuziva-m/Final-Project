"""
ocr/grid.py -- read a ruled table into the page's own columns, with no internet.

Record books are usually ruled: lines drawn down the page between columns,
and printed or drawn lines across it between rows. The fixed Date / Item /
Qty / Price / Total reader in ocr/fusion.py ignores that layout and guesses
columns from the text alone, so a page with other columns (a seed register,
a stock book) comes out empty or scrambled.

This module finds the ruled lines and puts every word EasyOCR reads into the
cell it sits in, so the table keeps its own columns and headings:

    grid = find_grid(image)                      # None if the page has no ruled columns
    table, words = read_table(grid, image, reader)
    # table: {"columns": [...], "rows": [[...], ...], "unsure": [[row, col], ...]}

Lines are found with OpenCV morphology and fitted as straight lines, so a
photo taken at a slight angle still works. EasyOCR's text boxes are cut where
a column line runs through them before the text is recognised, so writing
that runs up to a line is not merged with the next column.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np

# More columns than this is not a ledger page (likely a printed form or a photo
# of something else): leave it to the other readers.
MAX_COLUMNS = 30
# Words EasyOCR is less sure of than this are shaded for checking.
UNSURE_BELOW = 0.4
# Leading rows that are nearly all words (a heading is often written over two
# lines) are taken as the column headings.
MAX_HEADING_ROWS = 3
# On a ruled page, find fainter writing than EasyOCR's defaults do: lone "0"s
# in a stock column were missed otherwise. (Tested on a ruled register: more
# cells read right, no time lost.)
DETECT_TEXT_THRESHOLD, DETECT_LOW_TEXT, DETECT_MIN_SIZE = 0.5, 0.3, 10

# Letters OCR reads in place of handwritten digits. Only used in columns that
# are mostly numbers (amounts, quantities, dates), and only where the result is
# a number: "1S,000" -> "15,000", "Sooo" -> "5000", "1L.I2.17" -> "11.12.17".
_NUMBER_LOOKALIKES = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "i": "1", "L": "1", "|": "1",
                                    "!": "1", "Z": "2", "z": "2", "S": "5", "s": "5", "B": "8"})
_NUMBERISH = re.compile(r"[\d.,/:-]+")


@dataclass
class Line:
    """A straight ruled line. Vertical: x = slope * y + offset; horizontal: y = slope * x + offset."""

    slope: float
    offset: float

    def at(self, t: float) -> float:
        return self.slope * t + self.offset


@dataclass
class Grid:
    columns: list[Line]  # lines down the page, left to right
    rows: list[Line]     # lines across the page, top to bottom (may be empty)

    def column_of(self, x: float, y: float) -> int:
        return sum(1 for line in self.columns if line.at(y) < x)

    def row_of(self, x: float, y: float) -> int:
        return sum(1 for line in self.rows if line.at(x) < y)

    def tilt(self) -> float:
        """How much text rows rise or fall per pixel across, from the ruled lines."""
        if self.rows:
            return float(np.median([line.slope for line in self.rows]))
        return -float(np.median([line.slope for line in self.columns]))


# ---------------------------------------------------------------------------
# Finding the ruled lines
# ---------------------------------------------------------------------------


def _ink(gray: np.ndarray) -> np.ndarray:
    """Dark marks (writing and ruled lines, including faint printed ones) as white on black."""
    block = max(15, (min(gray.shape) // 60) | 1)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, 12)


def _fit(along: np.ndarray, pos: np.ndarray) -> Line:
    if len(along) > 4000:
        pick = np.linspace(0, len(along) - 1, 4000).astype(int)
        along, pos = along[pick], pos[pick]
    slope, offset = np.polyfit(along.astype(float), pos.astype(float), 1)
    return Line(float(slope), float(offset))


def _find_lines(ink: np.ndarray, vertical: bool, min_span: float) -> list[Line]:
    """Long straight lines in one direction, as fitted Lines sorted by position.

    *min_span* is the fraction of the page a line (all its pieces together)
    must cover to count.
    """
    h, w = ink.shape
    length, across = (h, w) if vertical else (w, h)
    k = max(20, length // 40)       # longer than any handwritten stroke
    widen = max(5, across // 250)   # so a slightly tilted line still has long straight runs
    along_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k) if vertical else (k, 1))
    widen_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (widen, 1) if vertical else (1, widen))

    mask = cv2.dilate(ink, widen_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, along_kernel)
    # Join pieces broken where writing crosses the line or the pen skipped.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, along_kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    segments = []
    for i in range(1, n):
        x, y, bw, bh, _area = stats[i]
        span, thick = (bh, bw) if vertical else (bw, bh)
        if span < 1.5 * k or thick > max(widen + 12, 0.15 * span):
            continue  # a short stroke or a blob, not part of a ruled line
        ys, xs = np.nonzero(labels[y:y + bh, x:x + bw] == i)
        along, pos = (ys + y, xs + x) if vertical else (xs + x, ys + y)
        segments.append((span, along, pos))

    # Pieces of the same line (and the two sides of a double-drawn line) are
    # joined: longest first, so each line's slope comes from its longest piece.
    segments.sort(key=lambda s: -s[0])
    tolerance = max(8, across * 0.008)
    clusters: list[dict] = []
    for span, along, pos in segments:
        mid, here = float(along.mean()), float(pos.mean())
        for cluster in clusters:
            if abs(cluster["line"].at(mid) - here) < tolerance:
                cluster["span"] += span
                cluster["along"] = np.concatenate([cluster["along"], along])
                cluster["pos"] = np.concatenate([cluster["pos"], pos])
                cluster["line"] = _fit(cluster["along"], cluster["pos"])
                break
        else:
            clusters.append({"span": span, "along": along, "pos": pos, "line": _fit(along, pos)})

    lines = [c["line"] for c in clusters if c["span"] >= min_span * length and abs(c["line"].slope) < 0.2]
    return sorted(lines, key=lambda line: line.at(length / 2))


def find_grid(image: np.ndarray) -> Grid | None:
    """The ruled columns (and rows, if any) of *image*, or None if it has no ruled columns."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ink = _ink(gray)
    columns = _find_lines(ink, vertical=True, min_span=0.25)
    if len(columns) < 2 or len(columns) > MAX_COLUMNS:
        return None
    rows = _find_lines(ink, vertical=False, min_span=0.3)
    return Grid(columns=columns, rows=rows)


# ---------------------------------------------------------------------------
# Reading the words into cells
# ---------------------------------------------------------------------------


def _split_at_columns(box: list[int], grid: Grid) -> list[list[int]]:
    """Cut an EasyOCR text box ``[x_min, x_max, y_min, y_max]`` where column lines run through it."""
    x0, x1, y0, y1 = box
    height = max(1, y1 - y0)
    cy = (y0 + y1) / 2
    cuts = sorted(x for x in (line.at(cy) for line in grid.columns) if x0 + 0.3 * height < x < x1 - 0.3 * height)
    if not cuts:
        return [box]
    pad = max(2, int(0.08 * height))  # leave the ruled line itself out of each piece
    edges = [(x0, 0)] + [(c, pad) for c in cuts] + [(x1, 0)]
    pieces = []
    for (a, pad_a), (b, pad_b) in zip(edges, edges[1:]):
        left, right = int(round(a + pad_a)), int(round(b - pad_b))
        if right - left >= 0.4 * height:
            pieces.append([left, right, y0, y1])
    return pieces


def _bands(values: list[float], gap: float) -> list[int]:
    """Group numbers that sit close together; the band index of each value, in order."""
    band_of = [0] * len(values)
    band, last = -1, None
    for i in sorted(range(len(values)), key=values.__getitem__):
        if last is None or values[i] - last > gap:
            band += 1
        band_of[i] = band
        last = values[i]
    return band_of


def _word_boxes(words: list) -> list[dict]:
    """EasyOCR results ``(points, text, confidence)`` as centre, height, text and confidence."""
    out = []
    for points, text, conf in words:
        text = str(text).strip().strip("|[]").strip()  # bits of a ruled line at the edge
        if not text:
            continue
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        out.append({"x": sum(xs) / len(xs), "y": sum(ys) / len(ys), "h": max(ys) - min(ys) or 1.0,
                    "text": text, "conf": float(conf)})
    return out


def _join(boxes: list[dict], tilt: float) -> str:
    """Words in reading order: line by line, then left to right."""
    if not boxes:
        return ""
    height = float(np.median([b["h"] for b in boxes]))
    line_of = _bands([b["y"] - tilt * b["x"] for b in boxes], 0.6 * height)
    order = sorted(range(len(boxes)), key=lambda i: (line_of[i], boxes[i]["x"]))
    return " ".join(boxes[i]["text"] for i in order)


def text_lines(words: list, tilt: float = 0.0) -> list[str]:
    """The lines of text EasyOCR read, top to bottom -- for when no table came out."""
    boxes = _word_boxes(words)
    if not boxes:
        return []
    height = float(np.median([b["h"] for b in boxes]))
    line_of = _bands([b["y"] - tilt * b["x"] for b in boxes], 0.6 * height)
    lines: dict[int, list[dict]] = {}
    for i, box in enumerate(boxes):
        lines.setdefault(line_of[i], []).append(box)
    return [" ".join(b["text"] for b in sorted(lines[n], key=lambda b: b["x"])) for n in sorted(lines)]


def _mostly_words(row: list[str]) -> bool:
    text = "".join(row).replace(" ", "")
    return sum(ch.isdigit() for ch in text) <= 0.1 * len(text)


def _is_number_column(cells: list[str]) -> bool:
    filled = [c.replace(" ", "") for c in cells if c]
    numeric = sum(1 for c in filled if 2 * sum(ch.isdigit() for ch in c) >= len(c))
    return len(filled) >= 2 and numeric >= 0.6 * len(filled)


def _fix_number(cell: str) -> str:
    """Swap look-alike letters for digits in the number-like words of *cell*."""
    has_digits = any(ch.isdigit() for ch in cell)
    words = []
    for word in cell.split():
        fixed = word.translate(_NUMBER_LOOKALIKES)
        if fixed != word and _NUMBERISH.fullmatch(fixed) and (has_digits or len(word) >= 3):
            word = fixed
        words.append(word)
    return " ".join(words)


def words_to_table(grid: Grid, words: list) -> dict | None:
    """Place recognised *words* into the cells of *grid*.

    Returns ``{"columns", "rows", "unsure"}`` or None when the words don't
    make a table (fewer than two columns with writing, or no rows under the
    headings).
    """
    boxes = _word_boxes(words)
    if not boxes:
        return None
    tilt = grid.tilt()
    height = float(np.median([b["h"] for b in boxes]))

    # Rows: the ruled lines when they are close enough to be one per line of
    # writing (lined paper); otherwise group the writing into lines.
    rows_by_rule = False
    if len(grid.rows) >= 3:
        mid_x = float(np.median([b["x"] for b in boxes]))
        spacing = float(np.median(np.diff([line.at(mid_x) for line in grid.rows])))
        rows_by_rule = spacing <= 3 * height
    if rows_by_rule:
        row_keys = [grid.row_of(b["x"], b["y"]) for b in boxes]
    else:
        row_keys = _bands([b["y"] - tilt * b["x"] for b in boxes], 0.6 * height)

    cells: dict[tuple[int, int], list[dict]] = {}
    for key, box in zip(row_keys, boxes):
        cells.setdefault((key, grid.column_of(box["x"], box["y"])), []).append(box)

    row_ids = sorted({r for r, _ in cells})
    col_ids = sorted({c for _, c in cells})
    if len(col_ids) < 2:
        return None

    grid_text = [[_join(cells.get((r, c), []), tilt) for c in col_ids] for r in row_ids]
    doubtful = [[any(b["conf"] < UNSURE_BELOW for b in cells.get((r, c), [])) for c in col_ids] for r in row_ids]

    heading_rows = 0
    for row in grid_text[:MAX_HEADING_ROWS]:
        if not _mostly_words(row):
            break
        heading_rows += 1
    if heading_rows >= len(grid_text):
        heading_rows = 1 if len(grid_text) > 1 else 0

    columns = [" ".join(row[c] for row in grid_text[:heading_rows] if row[c]).strip() or f"Column {c + 1}"
               for c in range(len(col_ids))]
    rows = grid_text[heading_rows:]
    if not rows:
        return None
    doubtful = doubtful[heading_rows:]
    for c in range(len(col_ids)):
        if _is_number_column([row[c] for row in rows]):
            for r, row in enumerate(rows):
                fixed = _fix_number(row[c])
                if fixed != row[c]:
                    row[c] = fixed
                    doubtful[r][c] = True  # a guessed digit: worth a look
    unsure = [[r, c] for r, flags in enumerate(doubtful) for c, flag in enumerate(flags) if flag]
    return {"columns": columns, "rows": rows, "unsure": unsure}


def read_words(reader, image: np.ndarray, grid: Grid | None = None) -> list:
    """EasyOCR's ``(points, text, confidence)`` for the colour (BGR) *image*.

    The same steps as ``reader.readtext``, given the colour and grey images
    it would make from a file, with one addition: with a *grid*, text boxes
    are cut where column lines run through them before the text is read.
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if grid is None:
        horizontal, free = reader.detect(rgb, reformat=False)
    else:
        horizontal, free = reader.detect(rgb, reformat=False, text_threshold=DETECT_TEXT_THRESHOLD,
                                         low_text=DETECT_LOW_TEXT, min_size=DETECT_MIN_SIZE)
    horizontal, free = horizontal[0], free[0]
    if grid is not None:
        horizontal = [piece for box in horizontal for piece in _split_at_columns([int(v) for v in box], grid)]
    if not horizontal and not free:
        return []
    return reader.recognize(grey, horizontal_list=horizontal, free_list=free, detail=1, paragraph=False,
                            reformat=False)


def read_table(grid: Grid, image: np.ndarray, reader) -> tuple[dict | None, list]:
    """Read *image* with EasyOCR *reader* into the cells of *grid*.

    Returns ``(table or None, words)``; the words are returned too so a
    caller can fall back to another reader without reading the page twice.
    """
    words = read_words(reader, image, grid)
    return words_to_table(grid, words), words
