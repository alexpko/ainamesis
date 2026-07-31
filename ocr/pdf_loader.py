"""
pdf_loader.py
Converts PDF pages to PIL Images for downstream OCR processing.
Falls back to direct image pass-through for JPEG/PNG inputs.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator

from PIL import Image

logger = logging.getLogger(__name__)

# Optional heavy dependency — imported lazily so the rest of the module
# works even when pdf2image / poppler are not installed.
_pdf2image_available = False
try:
    from pdf2image import convert_from_path  # type: ignore
    _pdf2image_available = True
except ImportError:
    logger.warning(
        "pdf2image not installed. PDF support is disabled. "
        "Install it with: pip install pdf2image"
    )

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def load_document(file_path: str | Path) -> Generator[Image.Image, None, None]:
    """
    Yield PIL Image objects for every page/frame of *file_path*.

    Parameters
    ----------
    file_path:
        Absolute or relative path to a PDF, JPG, or PNG file.

    Yields
    ------
    PIL.Image.Image
        One image per document page.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    FileNotFoundError
        If the file does not exist.
    RuntimeError
        If PDF loading is attempted without pdf2image installed.
    """
    path = Path(file_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    ext = path.suffix.lower()

    if ext == ".pdf":
        yield from _load_pdf(path)
    elif ext in SUPPORTED_IMAGE_EXTENSIONS:
        yield from _load_image(path)
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: .pdf, {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}"
        )


def _load_pdf(path: Path) -> Generator[Image.Image, None, None]:
    if not _pdf2image_available:
        raise RuntimeError(
            "pdf2image is required to process PDF files. "
            "Install it with: pip install pdf2image"
        )
    logger.info("Loading PDF: %s", path)
    pages = convert_from_path(str(path), dpi=300)
    logger.info("PDF has %d page(s): %s", len(pages), path.name)
    for i, page in enumerate(pages, start=1):
        logger.debug("Yielding PDF page %d of %d", i, len(pages))
        yield page


def _load_image(path: Path) -> Generator[Image.Image, None, None]:
    logger.info("Loading image: %s", path)
    img = Image.open(path).convert("RGB")
    yield img
