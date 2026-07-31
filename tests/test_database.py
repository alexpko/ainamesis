"""
test_database.py
Unit tests for the database module.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from database.models import DocumentRecord, ExtractionRecord, TranscriptionRecord
from database.sqlite import AInamesisDB


@pytest.fixture
def db(tmp_path):
    return AInamesisDB(db_path=tmp_path / "test.db")


class TestDocumentCRUD:
    def test_save_and_retrieve(self, db):
        record = DocumentRecord(
            file_path="/tmp/test.pdf",
            file_type="pdf",
            ocr_text="Patient complains of chest pain.",
        )
        doc_id = db.save_document(record)
        assert doc_id == 1

        fetched = db.get_document(doc_id)
        assert fetched is not None
        assert fetched.file_path == "/tmp/test.pdf"
        assert fetched.ocr_text == "Patient complains of chest pain."
        assert fetched.processed is False

    def test_mark_processed(self, db):
        doc_id = db.save_document(DocumentRecord(file_path="x.pdf", file_type="pdf"))
        db.mark_processed(doc_id)
        fetched = db.get_document(doc_id)
        assert fetched.processed is True

    def test_list_documents_empty(self, db):
        assert db.list_documents() == []

    def test_list_documents_multiple(self, db):
        db.save_document(DocumentRecord(file_path="a.pdf", file_type="pdf"))
        db.save_document(DocumentRecord(file_path="b.png", file_type="png"))
        docs = db.list_documents()
        assert len(docs) == 2

    def test_get_nonexistent_returns_none(self, db):
        assert db.get_document(999) is None


class TestExtractionCRUD:
    def test_save_and_retrieve(self, db):
        doc_id = db.save_document(DocumentRecord(file_path="scan.pdf", file_type="pdf"))
        record = ExtractionRecord(
            document_id=doc_id,
            model_used="llama3",
            diagnosis=["Hypertension"],
            medications=[{"name": "Amlodipine", "dose": "5mg", "frequency": None, "route": None}],
            allergies=["Aspirin"],
            laboratory_results=[],
            timeline=[],
        )
        ext_id = db.save_extraction(record)
        assert ext_id == 1

        extractions = db.get_extractions_for_document(doc_id)
        assert len(extractions) == 1
        assert extractions[0].diagnosis == ["Hypertension"]
        assert extractions[0].allergies == ["Aspirin"]

    def test_multiple_extractions_per_document(self, db):
        doc_id = db.save_document(DocumentRecord(file_path="x.pdf", file_type="pdf"))
        db.save_extraction(ExtractionRecord(document_id=doc_id, model_used="llama3"))
        db.save_extraction(ExtractionRecord(document_id=doc_id, model_used="mistral"))
        extractions = db.get_extractions_for_document(doc_id)
        assert len(extractions) == 2

    def test_extractions_for_nonexistent_doc(self, db):
        assert db.get_extractions_for_document(999) == []


class TestTranscriptionCRUD:
    def test_save_and_retrieve(self, db):
        record = TranscriptionRecord(
            audio_path="/tmp/consult.mp3",
            transcript="Patient reports fatigue for 3 weeks.",
            language="en",
            model_used="small",
        )
        t_id = db.save_transcription(record)
        assert t_id == 1

        fetched = db.get_transcription(t_id)
        assert fetched is not None
        assert fetched.transcript == "Patient reports fatigue for 3 weeks."
        assert fetched.language == "en"

    def test_get_nonexistent_transcription(self, db):
        assert db.get_transcription(999) is None


class TestDatabaseModels:
    def test_document_record_defaults(self):
        r = DocumentRecord(file_path="x.pdf", file_type="pdf")
        assert r.processed is False
        assert r.ocr_text is None
        assert isinstance(r.created_at, datetime)

    def test_extraction_record_defaults(self):
        r = ExtractionRecord(document_id=1, model_used="llama3")
        assert r.diagnosis == []
        assert r.medications == []
        assert r.allergies == []

    def test_transcription_record_defaults(self):
        r = TranscriptionRecord(audio_path="x.mp3", transcript="hello")
        assert r.model_used == "small"
        assert r.language is None
