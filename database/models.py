"""
models.py
Pydantic + SQLite-compatible data models for aInamnesis.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class DocumentRecord(BaseModel):
    """Represents a medical document ingested into the system."""
    id: Optional[int] = None
    file_path: str
    file_type: str          # "pdf" | "jpg" | "png" | "audio"
    ocr_text: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    processed: bool = False


class ExtractionRecord(BaseModel):
    """Links a document to its LLM-produced extraction."""
    id: Optional[int] = None
    document_id: int
    model_used: str
    diagnosis: List[str] = Field(default_factory=list)
    medications: List[dict] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    laboratory_results: List[dict] = Field(default_factory=list)
    timeline: List[dict] = Field(default_factory=list)
    raw_json: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)


class TranscriptionRecord(BaseModel):
    """Audio transcription record."""
    id: Optional[int] = None
    document_id: Optional[int] = None
    audio_path: str
    transcript: str
    language: Optional[str] = None
    model_used: str = "small"
    created_at: datetime = Field(default_factory=datetime.utcnow)
