"""
preprocessing/clean.py — OpenCV cleanup pipeline for phone-captured records.

Usage (batch):
    python -m preprocessing.clean                        # data/samples/ -> data/processed/
    python -m preprocessing.clean --input dir --output dir

Usage (library):
    from preprocessing.clean import preprocess
    cleaned = preprocess(image)                          # numpy array in, numpy array out
    cleaned = preprocess(image, save_path="out.png")

Drop any image into the input folder and it flows through without code changes.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

# Supported extensions for batch discovery.
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Individual pipeline steps — each is independently testable.
# ---------------------------------------------------------------------------


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert to single-channel grayscale.

    Phone photos are RGB/RGBA. All subsequent steps (denoising, thresholding,
    deskew angle detection) work on intensity only, so conversion happens first.
    Already-grayscale inputs (2-D arrays or single-channel 3-D arrays) pass
    through unchanged to make each step idempotent.
    """
    if image.ndim == 2:
        return image.copy()
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0].copy()
    if image.ndim == 3 and image.shape[2] == 4:  # RGBA
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def deskew(image: np.ndarray, skew_threshold_deg: float = 0.5) -> np.ndarray:
    """Detect and correct document skew introduced by hand-held phone capture.

    Phone photos are rarely perfectly level — even a 2–3° tilt causes
    Tesseract/EasyOCR to misread characters on long text lines.

    Algorithm:
      1. Binary-threshold the grayscale image to isolate text pixels.
      2. Find the minimum-area bounding rectangle of all non-zero (text) pixels
         via cv2.minAreaRect — its angle encodes the dominant text baseline.
      3. Rotate the image by the negative of that angle around its centre.

    The ``skew_threshold_deg`` guard skips the (slightly lossy) rotation for
    images that are already nearly level, avoiding unnecessary interpolation.
    """
    gray = image if image.ndim == 2 else to_grayscale(image)

    # Threshold to separate text from background, suppressing noise.
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # All non-zero pixels (text) as a point set for minAreaRect.
    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] < 5:
        # Too few text pixels to estimate angle reliably; return as-is.
        return image.copy()

    # minAreaRect returns angle in (-90, 0]; convert to a signed tilt.
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle  # long edge is horizontal

    if abs(angle) < skew_threshold_deg:
        return image.copy()

    h, w = image.shape[:2]
    centre = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(centre, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def denoise(image: np.ndarray, h: int = 10) -> np.ndarray:
    """Remove sensor and JPEG compression noise with Non-Local Means denoising.

    Phone camera sensors introduce luminance noise, especially in low light.
    NLM (``cv2.fastNlMeansDenoising``) averages similar patches across the
    image — strong enough to suppress noise but gentler on text edges than a
    Gaussian blur, which would bleed strokes together and hurt OCR accuracy.

    ``h`` controls filter strength (higher = more smoothing, more detail loss).
    10 is a reasonable default for document shots; reduce if fine strokes blur.
    """
    gray = image if image.ndim == 2 else to_grayscale(image)
    return cv2.fastNlMeansDenoising(gray, h=h, templateWindowSize=7, searchWindowSize=21)


def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    """Equalise local contrast with CLAHE (Contrast Limited Adaptive Histogram Equalisation).

    Phone photos of paper documents have uneven illumination — bright near a
    window, dark in corners, or shadowed by the phone itself. Global histogram
    equalisation over-brightens already-bright regions. CLAHE divides the image
    into tiles and equalises each independently, capped by ``clip_limit`` to
    prevent noise amplification in flat regions.

    ``tile_grid`` sets the tile size (tile_grid × tile_grid). Smaller tiles
    respond to finer illumination gradients but amplify noise more.
    """
    gray = image if image.ndim == 2 else to_grayscale(image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    return clahe.apply(gray)


def binarise(image: np.ndarray) -> np.ndarray:
    """Threshold to a clean black-on-white binary image.

    OCR engines work best on pure black text on a white background. Adaptive
    (Gaussian) thresholding computes a local threshold for each pixel from its
    neighbourhood, handling residual illumination gradients that CLAHE did not
    fully correct. Otsu's method is a good fallback for images that are already
    evenly lit.

    The adaptive threshold with a 31-pixel neighbourhood and C=10 offset is
    tuned for typical A5–A4 document captures at 1–3 MP resolution; adjust
    ``blockSize`` for very high-resolution scans.
    """
    gray = image if image.ndim == 2 else to_grayscale(image)
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=10,
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def preprocess(image: np.ndarray, save_path: str | os.PathLike | None = None) -> np.ndarray:
    """Run the full cleanup pipeline and return the cleaned image.

    Chain: grayscale → deskew → denoise → enhance_contrast → binarise.

    Parameters
    ----------
    image:
        Input image as a NumPy array (BGR, BGRA, or grayscale). Must not be None.
    save_path:
        If given, the cleaned image is written to this path (parent dirs are
        created automatically). The return value is the cleaned array regardless.

    Returns
    -------
    np.ndarray
        Single-channel (grayscale) cleaned binary image.
    """
    if image is None or image.size == 0:
        raise ValueError("preprocess() received an empty or None image.")

    result = to_grayscale(image)
    result = deskew(result)
    result = denoise(result)
    result = enhance_contrast(result)
    result = binarise(result)

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), result)

    return result


# ---------------------------------------------------------------------------
# Debug view — saves each intermediate step
# ---------------------------------------------------------------------------


def preprocess_debug(image_path: str | os.PathLike, debug_dir: str | os.PathLike = "data/debug") -> np.ndarray:
    """Run the pipeline and save every intermediate step to ``debug_dir``.

    Useful for diagnosing which step introduces or removes an artefact. Each
    saved file is prefixed with the source filename so multiple runs don't
    overwrite each other.

    Parameters
    ----------
    image_path:
        Path to the source image file.
    debug_dir:
        Directory where intermediate images are saved. Created if absent.

    Returns
    -------
    np.ndarray
        The final cleaned image (same as ``preprocess()`` output).
    """
    image_path = Path(image_path)
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    src = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if src is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    stem = image_path.stem

    def _save(tag: str, img: np.ndarray) -> None:
        out_path = debug_dir / f"{stem}_{tag}.png"
        cv2.imwrite(str(out_path), img)
        print(f"  [debug] saved {out_path}")

    _save("0_original", src)

    gray = to_grayscale(src)
    _save("1_grayscale", gray)

    deskewed = deskew(gray)
    _save("2_deskew", deskewed)

    denoised = denoise(deskewed)
    _save("3_denoise", denoised)

    contrasted = enhance_contrast(denoised)
    _save("4_contrast", contrasted)

    binary = binarise(contrasted)
    _save("5_binarise", binary)

    return binary


# ---------------------------------------------------------------------------
# Batch __main__
# ---------------------------------------------------------------------------


def _batch(input_dir: Path, output_dir: Path) -> None:
    """Process every supported image in ``input_dir`` into ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = [
        p for p in sorted(input_dir.iterdir())
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    ]

    if not sources:
        print(f"No images found in {input_dir} (supported: {sorted(_IMAGE_EXTS)})")
        return

    print(f"Processing {len(sources)} image(s) from '{input_dir}' -> '{output_dir}'")
    for src_path in sources:
        img = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"  [skip] could not read {src_path}")
            continue
        out_path = output_dir / (src_path.stem + "_clean.png")
        preprocess(img, save_path=out_path)
        print(f"  [ok]   {src_path.name} -> {out_path.name}")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess phone-captured record images for OCR.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Drop any image into INPUT_DIR and it flows through without code changes.",
    )
    parser.add_argument(
        "--input", default="data/samples",
        help="Folder containing source images (default: data/samples)",
    )
    parser.add_argument(
        "--output", default="data/processed",
        help="Folder for cleaned output images (default: data/processed)",
    )
    args = parser.parse_args()

    _batch(Path(args.input), Path(args.output))
