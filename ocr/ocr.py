"""
ocr.py
Multi-language OCR engine.
Primary: EasyOCR
Fallback: Tesseract (pytesseract)

Supports: PDF, JPG, PNG (and any format handled by pdf_loader).
Returns structured text with per-page confidence metadata.
Saves a plain-text log of every run to <logs_dir>/ocr_<timestamp>.log
"""
from __future__ import annotations

import logging
import os
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from PIL import Image

from ocr.pdf_loader import load_document

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
logging.basicConfig(format=_LOG_FORMAT, stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports — imported lazily to keep startup fast
# ---------------------------------------------------------------------------
_easyocr_available = False
_tesseract_available = False

try:
    import easyocr  # type: ignore
    _easyocr_available = True
except ImportError:
    logger.warning("easyocr not installed. Tesseract fallback will be used.")

try:
    import pytesseract  # type: ignore
    _tesseract_available = True
except ImportError:
    logger.warning("pytesseract not installed. OCR functionality is limited.")


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------
@dataclass
class PageResult:
    """OCR result for a single page / image."""
    page_number: int
    text: str
    confidence: Optional[float]   # 0.0–1.0, None when unavailable
    engine_used: str              # "easyocr" | "tesseract" | "none"


@dataclass
class OCRResult:
    """Aggregate result for a whole document."""
    source_path: str
    pages: List[PageResult] = field(default_factory=list)
    log_file: Optional[str] = None

    @property
    def full_text(self) -> str:
        """Concatenated plain text for all pages."""
        return "\n\n".join(p.text for p in self.pages if p.text.strip())

    @property
    def average_confidence(self) -> Optional[float]:
        confidences = [p.confidence for p in self.pages if p.confidence is not None]
        return sum(confidences) / len(confidences) if confidences else None


# ---------------------------------------------------------------------------
# Engine wrapper
# ---------------------------------------------------------------------------
class OCREngine:
    """
    Stateful OCR engine that can be reused across multiple documents.

    Parameters
    ----------
    languages:
        List of BCP-47 / EasyOCR language codes.
        Defaults to ['en'] (English).
        Example multi-language: ['en', 'de', 'fr', 'pt']
    gpu:
        Whether to enable GPU acceleration for EasyOCR.
    logs_dir:
        Directory where per-run log files are stored.
        Defaults to <cwd>/logs/ocr/
    prefer_easyocr:
        When both engines are available, use EasyOCR first (default True).
    tesseract_lang:
        Tesseract language string (e.g. "eng+deu"). If None, derived from
        *languages* automatically.
    """

    def __init__(
        self,
        languages: List[str] | None = None,
        gpu: bool = False,
        logs_dir: str | Path | None = None,
        prefer_easyocr: bool = True,
        tesseract_lang: str | None = None,
    ) -> None:
        self.languages = languages or ["en"]
        self.gpu = gpu
        self.logs_dir = Path(logs_dir) if logs_dir else Path.cwd() / "logs" / "ocr"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.prefer_easyocr = prefer_easyocr
        self.tesseract_lang = tesseract_lang or self._build_tesseract_lang()

        # Initialise EasyOCR reader once — expensive operation
        self._easyocr_reader = None
        if _easyocr_available and prefer_easyocr:
            self._init_easyocr()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process_file(self, file_path: str | Path) -> OCRResult:
        """
        Run OCR on *file_path* (PDF, JPG, or PNG).

        Returns
        -------
        OCRResult
            Structured result with per-page text, confidence, and engine used.
        """
        file_path = Path(file_path)
        logger.info("Starting OCR: %s", file_path.name)

        result = OCRResult(source_path=str(file_path))
        log_lines: list[str] = [
            f"=== aInamnesis OCR Log ===",
            f"File     : {file_path}",
            f"Started  : {datetime.now().isoformat()}",
            f"Languages: {self.languages}",
            "",
        ]

        for page_num, image in enumerate(load_document(file_path), start=1):
            page_result = self._process_image(image, page_num)
            result.pages.append(page_result)
            log_lines.append(
                f"--- Page {page_num} | engine={page_result.engine_used} "
                f"| confidence={page_result.confidence} ---"
            )
            log_lines.append(page_result.text)
            log_lines.append("")
            logger.info(
                "Page %d done | engine=%s | chars=%d",
                page_num,
                page_result.engine_used,
                len(page_result.text),
            )

        log_lines.append(f"Finished : {datetime.now().isoformat()}")
        result.log_file = self._save_log("\n".join(log_lines), file_path.stem)

        logger.info(
            "OCR complete: %d page(s), avg_confidence=%.3f, log=%s",
            len(result.pages),
            result.average_confidence or 0.0,
            result.log_file,
        )
        return result

    def process_image(self, image: Image.Image, page_number: int = 1) -> PageResult:
        """Run OCR on a pre-loaded PIL Image."""
        return self._process_image(image, page_number)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _init_easyocr(self) -> None:
        logger.info("Initialising EasyOCR reader (languages=%s, gpu=%s)…", self.languages, self.gpu)
        try:
            self._easyocr_reader = easyocr.Reader(self.languages, gpu=self.gpu)
            logger.info("EasyOCR reader ready.")
        except Exception as exc:
            logger.error("Failed to initialise EasyOCR: %s. Falling back to Tesseract.", exc)
            self._easyocr_reader = None

    def _process_image(self, image: Image.Image, page_num: int) -> PageResult:
        """Dispatch to the appropriate engine."""
        # Attempt EasyOCR first when preferred and available
        if self.prefer_easyocr and self._easyocr_reader is not None:
            return self._run_easyocr(image, page_num)
        # Tesseract fallback
        if _tesseract_available:
            return self._run_tesseract(image, page_num)
        # Last resort: try to (re-)initialise EasyOCR even when not preferred
        if _easyocr_available and self._easyocr_reader is None:
            self._init_easyocr()
            if self._easyocr_reader is not None:
                return self._run_easyocr(image, page_num)

        logger.error("No OCR engine available for page %d.", page_num)
        return PageResult(
            page_number=page_num,
            text="",
            confidence=None,
            engine_used="none",
        )

    def _run_easyocr(self, image: Image.Image, page_num: int) -> PageResult:
        """Run EasyOCR on a single PIL image."""
        try:
            import numpy as np  # type: ignore
            img_array = np.array(image)
            detections = self._easyocr_reader.readtext(img_array, detail=1)
            # detections: list of ([bbox], text, confidence)
            texts = []
            confidences = []
            for _bbox, text, conf in detections:
                texts.append(text)
                confidences.append(conf)

            full_text = " ".join(texts)
            avg_conf = sum(confidences) / len(confidences) if confidences else None
            return PageResult(
                page_number=page_num,
                text=full_text,
                confidence=avg_conf,
                engine_used="easyocr",
            )
        except Exception as exc:
            logger.warning("EasyOCR failed on page %d (%s). Trying Tesseract…", page_num, exc)
            if _tesseract_available:
                return self._run_tesseract(image, page_num)
            return PageResult(page_number=page_num, text="", confidence=None, engine_used="none")

    def _run_tesseract(self, image: Image.Image, page_num: int) -> PageResult:
        """Run pytesseract on a single PIL image."""
        try:
            data = pytesseract.image_to_data(
                image,
                lang=self.tesseract_lang,
                output_type=pytesseract.Output.DICT,
            )
            words = []
            confidences = []
            for word, conf in zip(data["text"], data["conf"]):
                try:
                    conf_int = int(conf)
                except (ValueError, TypeError):
                    continue
                if conf_int > 0 and word.strip():
                    words.append(word)
                    confidences.append(conf_int / 100.0)

            full_text = " ".join(words)
            avg_conf = sum(confidences) / len(confidences) if confidences else None
            return PageResult(
                page_number=page_num,
                text=full_text,
                confidence=avg_conf,
                engine_used="tesseract",
            )
        except Exception as exc:
            logger.error("Tesseract failed on page %d: %s", page_num, exc)
            return PageResult(page_number=page_num, text="", confidence=None, engine_used="none")

    def _save_log(self, content: str, stem: str) -> str:
        """Persist log content and return the path as string."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = self.logs_dir / f"ocr_{stem}_{timestamp}.log"
        try:
            log_path.write_text(content, encoding="utf-8")
            logger.debug("OCR log saved: %s", log_path)
        except OSError as exc:
            logger.error("Could not write OCR log: %s", exc)
        return str(log_path)

    def _build_tesseract_lang(self) -> str:
        """Convert EasyOCR language codes to Tesseract lang string."""
        _EASYOCR_TO_TESSERACT: dict[str, str] = {
            "en": "eng",
            "de": "deu",
            "fr": "fra",
            "pt": "por",
            "es": "spa",
            "it": "ita",
            "nl": "nld",
            "ru": "rus",
            "zh": "chi_sim",
            "ja": "jpn",
            "ko": "kor",
            "ar": "ara",
        }
        tess_codes = [_EASYOCR_TO_TESSERACT.get(lang, lang) for lang in self.languages]
        return "+".join(tess_codes) or "eng"
