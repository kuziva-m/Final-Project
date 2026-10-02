"""
preprocessing/pdf_pages.py -- render the pages of an uploaded PDF to images.

Scanned record books often arrive as one large PDF. Each page is rendered to
a PNG so it can go through the same per-image pipeline as a phone photo
(preprocess, then Claude or Tesseract + EasyOCR), and so no single request
to the Claude API carries the whole file.
"""

from __future__ import annotations

from pathlib import Path

_MAX_EDGE_PX = 3000


class PdfError(ValueError):
    """The PDF could not be opened or has no pages."""


def render_pages(pdf_path: Path, out_dir: Path, stem: str, max_pages: int, dpi: int = 200) -> tuple[list[Path], int]:
    """Render up to *max_pages* pages of *pdf_path* to ``{stem}_p{n}.png``.

    Returns ``(page_image_paths, total_page_count)``. 200 dpi keeps
    handwriting legible for Tesseract; the Claude reader downsizes anyway.
    """
    import pymupdf

    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception as exc:  # PyMuPDF raises several unrelated types for bad files
        raise PdfError("That PDF could not be opened. It may be damaged.") from exc

    with doc:
        if doc.needs_pass:
            raise PdfError("That PDF is password-protected. Remove the password and try again.")
        total = doc.page_count
        if total == 0:
            raise PdfError("That PDF has no pages.")

        paths = []
        for n in range(min(total, max_pages)):
            page = doc[n]
            # Cap the longest edge so a PDF declaring a giant page can't
            # exhaust memory.
            scale = min(dpi / 72, _MAX_EDGE_PX / max(page.rect.width, page.rect.height, 1))
            out = out_dir / f"{stem}_p{n + 1}.png"
            page.get_pixmap(matrix=pymupdf.Matrix(scale, scale)).save(str(out))
            paths.append(out)
    return paths, total

