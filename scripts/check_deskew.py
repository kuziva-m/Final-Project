"""scripts/check_deskew.py -- does deskew actually level a tilted page?

Tilts a level copy of the synthetic ledger by known angles, runs deskew, and
measures what tilt is left with an independent method (the angle whose
horizontal projection profile is sharpest). Run against the previous deskew
code as well, so the fix can be compared with the bug it replaces.

    python scripts/check_deskew.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing.clean import deskew, to_grayscale  # noqa: E402
from scripts.make_synthetic_ledger import generate  # noqa: E402


def previous_deskew(image: np.ndarray, skew_threshold_deg: float = 0.5) -> np.ndarray:
    """The deskew code before the fix, unchanged, for comparison."""
    gray = image if image.ndim == 2 else to_grayscale(image)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] < 5:
        return image.copy()
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < skew_threshold_deg:
        return image.copy()
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def remaining_tilt(gray: np.ndarray) -> float:
    """Independent estimate: the rotation that makes text rows sharpest."""
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    h, w = ink.shape
    best, best_score = 0.0, -1.0
    for a in np.arange(-20, 20.01, 0.1):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        score = float(np.var(cv2.warpAffine(ink, M, (w, h)).sum(axis=1)))
        if score > best_score:
            best, best_score = a, score
    return round(best, 1) + 0.0  # same sign as tilt(): positive = text slopes down to the right


def tilt(gray: np.ndarray, degrees: float) -> np.ndarray:
    """Tilt clockwise by *degrees* (positive = text slopes down to the right)."""
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -degrees, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=250)


def main() -> None:
    level_path = Path("data/debug/synthetic_ledger_level.png")
    # A level copy: same rows, noise and seed, without the notebook's tilt.
    import scripts.make_synthetic_ledger as gen
    original_rotate = Image.Image.rotate
    Image.Image.rotate = lambda self, *a, **k: self  # skip the 3 degree tilt for this copy
    try:
        gen.generate(level_path)
    finally:
        Image.Image.rotate = original_rotate
    level = cv2.cvtColor(cv2.imread(str(level_path)), cv2.COLOR_BGR2GRAY)
    pad = cv2.copyMakeBorder(level, 80, 80, 80, 80, cv2.BORDER_CONSTANT, value=250)

    print(f"{'tilt applied':>12} | {'left by previous code':>22} | {'left by fixed code':>19}")
    worst_old = worst_new = 0.0
    for applied in (-7.0, -3.0, -1.5, 1.5, 3.0, 7.0):
        tilted = tilt(pad, applied)
        old_left = remaining_tilt(previous_deskew(tilted))
        new_left = remaining_tilt(deskew(tilted))
        worst_old, worst_new = max(worst_old, abs(old_left)), max(worst_new, abs(new_left))
        print(f"{applied:>+11.1f}° | {old_left:>+21.1f}° | {new_left:>+18.1f}°")
    print(f"largest tilt left: previous code {worst_old:.1f}°, fixed code {worst_new:.1f}°")
    real = cv2.cvtColor(cv2.imread("data/samples/synthetic_ledger.png"), cv2.COLOR_BGR2GRAY)
    print(f"synthetic_ledger.png (the notebook's 3° tilt): {remaining_tilt(real):+.1f}° before, "
          f"{remaining_tilt(previous_deskew(real)):+.1f}° after previous code, "
          f"{remaining_tilt(deskew(real)):+.1f}° after fixed code")


if __name__ == "__main__":
    main()
