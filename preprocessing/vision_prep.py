"""
preprocessing/vision_prep.py -- prepare a page image for Claude's vision model.

Claude's image cost grows with pixel area, so every image is scaled to a fixed
pixel budget. Cropping away scanner borders and empty margins first means the
handwriting fills more of that budget -- larger, clearer text for the same
(or lower) cost. Faded ink is lifted with a gentle contrast boost. Unlike
preprocessing/clean.py (which binarises for Tesseract), colour and shading
are kept, since the vision model reads them well.
"""

from __future__ import annotations

import cv2
import numpy as np

# About what a full A4 page cost before cropping (1568 x 1109 px, ~2,300
# image tokens): pages never cost more than that, and cropped pages get
# sharper text instead of a bigger bill.
MAX_PIXELS = 1_740_000
# Longest edge the model accepts at full resolution.
MAX_EDGE = 2576


def _crop_to_paper(img: np.ndarray) -> np.ndarray:
    """Drop dark scanner/table background around the sheet, if there is any."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bright = cv2.threshold(cv2.GaussianBlur(gray, (9, 9), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    # Only crop when a clear sheet was found and it isn't already the whole image.
    if w * h < 0.25 * img.shape[0] * img.shape[1] or w * h > 0.97 * img.shape[0] * img.shape[1]:
        return img
    return img[y:y + h, x:x + w]


def _crop_to_writing(img: np.ndarray) -> np.ndarray:
    """Trim blank margins around the writing and ruled lines."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))  # drop speckle
    edge = max(2, int(0.01 * min(h, w)))
    ink[:edge, :] = ink[-edge:, :] = 0  # ignore shadows along the sheet edges
    ink[:, :edge] = ink[:, -edge:] = 0
    ys, xs = np.nonzero(ink)
    if len(xs) < 500:  # (nearly) blank page: nothing to crop to
        return img
    x0, x1 = np.percentile(xs, [0.5, 99.5]).astype(int)
    y0, y1 = np.percentile(ys, [0.5, 99.5]).astype(int)
    pad_x, pad_y = int(0.03 * w), int(0.03 * h)
    x0, x1 = max(0, x0 - pad_x), min(w, x1 + pad_x)
    y0, y1 = max(0, y0 - pad_y), min(h, y1 + pad_y)
    if (x1 - x0) * (y1 - y0) < 0.2 * w * h:  # suspiciously small: keep the page
        return img
    return img[y0:y1, x0:x1]


def _lift_faded_ink(img: np.ndarray) -> np.ndarray:
    """Gentle local contrast boost on lightness only (colours untouched)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _fit_budget(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(1.0, (MAX_PIXELS / (w * h)) ** 0.5, MAX_EDGE / max(w, h))
    if scale < 1.0:
        img = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    return img


def prepare_for_vision(img: np.ndarray) -> np.ndarray:
    """Crop to the sheet and its writing, lift faded ink, fit the pixel budget."""
    img = _crop_to_paper(img)
    img = _crop_to_writing(img)
    img = _lift_faded_ink(img)
    return _fit_budget(img)
