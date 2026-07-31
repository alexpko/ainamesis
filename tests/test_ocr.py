"""
test_ocr.py
Unit tests for the OCR module.
"""
from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def white_image():
    """1×1 white RGB image."""
    img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    return img


@pytest.fixture
def tmp_png(tmp_path, white_image):
    p = tmp_path / "test.png"
    white_image.save(p)
    return p


@pytest.fixture
def tmp_jpg(tmp_path, white_image):
    p = tmp_path / "test.jpg"
    white_image.save(p)
    return p


# ---------------------------------------------------------------------------
# pdf_loader tests
# ---------------------------------------------------------------------------

class TestPdfLoader:
    def test_unsupported_extension_raises(self, tmp_path):
        from ocr.pdf_loader import load_document
        bad_file = tmp_path / "test.docx"
        bad_file.write_bytes(b"dummy")
        with pytest.raises(ValueError, match="Unsupported file type"):
            list(load_document(bad_file))

    def test_file_not_found_raises(self, tmp_path):
        from ocr.pdf_loader import load_document
        with pytest.raises(FileNotFoundError):
            list(load_document(tmp_path / "nonexistent.png"))

    def test_loads_png(self, tmp_png):
        from ocr.pdf_loader import load_document
        pages = list(load_document(tmp_png))
        assert len(pages) == 1
        assert isinstance(pages[0], Image.Image)

    def test_loads_jpg(self, tmp_jpg):
        from ocr.pdf_loader import load_document
        pages = list(load_document(tmp_jpg))
        assert len(pages) == 1

    def test_pdf_raises_without_pdf2image(self, tmp_path, monkeypatch):
        import ocr.pdf_loader as pl
        monkeypatch.setattr(pl, "_pdf2image_available", False)
        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake")
        from ocr.pdf_loader import load_document
        with pytest.raises(RuntimeError, match="pdf2image"):
            list(load_document(fake_pdf))


# ---------------------------------------------------------------------------
# OCREngine tests
# ---------------------------------------------------------------------------

class TestOCREngine:
    def test_init_no_engines_still_creates(self, tmp_path, monkeypatch):
        """Engine should not crash on init even when neither OCR lib is available."""
        import ocr.ocr as ocr_module
        monkeypatch.setattr(ocr_module, "_easyocr_available", False)
        monkeypatch.setattr(ocr_module, "_tesseract_available", False)
        from ocr.ocr import OCREngine
        engine = OCREngine(logs_dir=tmp_path)
        assert engine is not None

    def test_process_image_returns_page_result_with_easyocr(self, white_image, tmp_path, monkeypatch):
        import ocr.ocr as ocr_module
        monkeypatch.setattr(ocr_module, "_easyocr_available", True)

        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([[0, 0], [10, 0], [10, 10], [0, 10]], "Diagnosis: Hypertension", 0.95)
        ]

        import numpy as np
        from ocr.ocr import OCREngine
        engine = OCREngine(languages=["en"], gpu=False, logs_dir=tmp_path)
        # Inject the mock reader directly — avoids needing easyocr installed
        engine._easyocr_reader = mock_reader
        result = engine.process_image(white_image, page_number=1)

        assert result.engine_used == "easyocr"
        assert "Hypertension" in result.text
        assert result.confidence == pytest.approx(0.95)

    def test_process_image_tesseract_fallback(self, white_image, tmp_path, monkeypatch):
        """
        pytesseract is imported lazily inside _run_tesseract, so we patch
        the engine method directly to validate the dispatch logic.
        """
        import ocr.ocr as ocr_module
        from ocr.ocr import OCREngine, PageResult

        monkeypatch.setattr(ocr_module, "_easyocr_available", False)
        monkeypatch.setattr(ocr_module, "_tesseract_available", True)

        engine = OCREngine(languages=["en"], prefer_easyocr=False, logs_dir=tmp_path)

        expected = PageResult(page_number=1, text="Glucose 5.4", confidence=0.90, engine_used="tesseract")
        monkeypatch.setattr(engine, "_run_tesseract", lambda img, pn: expected)

        result = engine._process_image(white_image, 1)

        assert result.engine_used == "tesseract"
        assert "Glucose" in result.text

    def test_process_file_saves_log(self, tmp_png, tmp_path, monkeypatch):
        import ocr.ocr as ocr_module
        monkeypatch.setattr(ocr_module, "_easyocr_available", True)
        monkeypatch.setattr(ocr_module, "_tesseract_available", False)

        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [([], "Test text", 0.8)]

        from ocr.ocr import OCREngine
        engine = OCREngine(languages=["en"], gpu=False, logs_dir=tmp_path)
        engine._easyocr_reader = mock_reader

        result = engine.process_file(tmp_png)
        assert result.log_file is not None
        assert Path(result.log_file).exists()

    def test_full_text_property(self, tmp_path, monkeypatch):
        from ocr.ocr import OCRResult, PageResult
        r = OCRResult(source_path="test.png")
        r.pages.append(PageResult(page_number=1, text="Hello", confidence=0.9, engine_used="easyocr"))
        r.pages.append(PageResult(page_number=2, text="World", confidence=0.85, engine_used="easyocr"))
        assert r.full_text == "Hello\n\nWorld"

    def test_average_confidence_empty(self):
        from ocr.ocr import OCRResult
        r = OCRResult(source_path="test.png")
        assert r.average_confidence is None

    def test_tesseract_lang_builder(self, tmp_path):
        from ocr.ocr import OCREngine
        engine = OCREngine(languages=["en", "de", "fr"], logs_dir=tmp_path)
        assert engine.tesseract_lang == "eng+deu+fra"
