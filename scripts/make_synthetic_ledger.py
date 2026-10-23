"""scripts/make_synthetic_ledger.py -- the synthetic test ledger from notebooks/spike_ocr_test.ipynb.

Renders the same six-row ledger (date, item, qty, price, total) with the same
mild noise, 3 degree tilt and seed as the notebook, so the pipeline has an input
whose ground truth is known exactly (data/ground_truth/synthetic_ledger*.txt/csv).

    python scripts/make_synthetic_ledger.py            # -> data/samples/synthetic_ledger.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROWS = [
    ["Date", "Item", "Qty", "Price", "Total"],
    ["2026-01-03", "Maize meal", "2", "3.50", "7.00"],
    ["2026-01-05", "Cooking oil", "1", "4.25", "4.25"],
    ["2026-01-09", "Sugar 2kg", "3", "2.10", "6.30"],
    ["2026-01-12", "Bread", "5", "1.15", "5.75"],
    ["2026-01-15", "Soap bar", "4", "0.90", "3.60"],
    ["2026-01-20", "Tea leaves", "1", "2.75", "2.75"],
]


def load_font(size: int) -> ImageFont.ImageFont:
    """The notebook's font list, plus matplotlib's copy of DejaVu Sans for Windows."""
    candidates = [
        "/usr/share/fonts/truetype/humor-sans/Humor-Sans.ttf",
        "Humor-Sans.ttf",
        "/usr/share/fonts/truetype/comic-sans/ComicSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "DejaVuSans.ttf",
    ]
    try:
        import matplotlib
        candidates.append(str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"))
    except ImportError:
        pass
    for path in candidates:
        try:
            font = ImageFont.truetype(path, size)
            print(f"Using font: {path}")
            return font
        except OSError:
            continue
    print("No TrueType font found -- falling back to PIL default bitmap font.")
    return ImageFont.load_default()


def generate(out_path: Path, seed: int = 42) -> Path:
    rng = np.random.default_rng(seed)
    W, H = 900, 460
    img = Image.new("RGB", (W, H), color=(252, 250, 244))
    draw = ImageDraw.Draw(img)
    font, header_font = load_font(26), load_font(28)
    col_x = [30, 210, 470, 580, 720]
    y0, row_h = 40, 56
    draw.text((30, 8), "Shop Ledger — January", fill=(20, 20, 20), font=header_font)
    for r, row in enumerate(ROWS):
        y = y0 + r * row_h
        for c, cell in enumerate(row):
            draw.text((col_x[c], y), str(cell), fill=(15, 15, 15), font=header_font if r == 0 else font)
        draw.line([(20, y + row_h - 12), (W - 20, y + row_h - 12)], fill=(200, 195, 180), width=1)
    arr = np.asarray(img).astype(np.float32)
    arr = np.clip(arr + rng.normal(0, 8.0, arr.shape), 0, 255).astype(np.uint8)
    img = Image.fromarray(arr).rotate(-3.0, resample=Image.BICUBIC, expand=True, fillcolor=(252, 250, 244))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"Saved synthetic ledger to: {out_path}  ({img.size[0]}x{img.size[1]})")
    return out_path


if __name__ == "__main__":
    generate(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/samples/synthetic_ledger.png"))
