"""
sqlite.py
SQLite persistence layer for aInamnesis.
All data stays local — no external database connections.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator, List, Optional

from database.models import DocumentRecord, ExtractionRecord, TranscriptionRecord

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path.cwd() / "data" / "ainamesis.db"

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path   TEXT    NOT NULL,
    file_type   TEXT    NOT NULL,
    ocr_text    TEXT,
    created_at  TEXT    NOT NULL,
    processed   INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS extractions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id         INTEGER NOT NULL REFERENCES documents(id),
    model_used          TEXT    NOT NULL,
    diagnosis           TEXT    NOT NULL DEFAULT '[]',
    medications         TEXT    NOT NULL DEFAULT '[]',
    allergies           TEXT    NOT NULL DEFAULT '[]',
    laboratory_results  TEXT    NOT NULL DEFAULT '[]',
    timeline            TEXT    NOT NULL DEFAULT '[]',
    raw_json            TEXT    NOT NULL DEFAULT '',
    created_at          TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS transcriptions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER REFERENCES documents(id),
    audio_path  TEXT    NOT NULL,
    transcript  TEXT    NOT NULL,
    language    TEXT,
    model_used  TEXT    NOT NULL DEFAULT 'small',
    created_at  TEXT    NOT NULL
);
"""


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------

class AInamesisDB:
    """
    Thin SQLite repository.  Thread-safe via WAL mode.

    Parameters
    ----------
    db_path:
        Path to the SQLite file.  Created automatically if it does not exist.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info("Database ready: %s", self.db_path)

    # ------------------------------------------------------------------
    # Context manager for connections
    # ------------------------------------------------------------------

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)

    # ------------------------------------------------------------------
    # Documents
    # ------------------------------------------------------------------

    def save_document(self, record: DocumentRecord) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO documents (file_path, file_type, ocr_text, created_at, processed)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    record.file_path,
                    record.file_type,
                    record.ocr_text,
                    _dt(record.created_at),
                    int(record.processed),
                ),
            )
            return cursor.lastrowid  # type: ignore[return-value]

    def get_document(self, doc_id: int) -> Optional[DocumentRecord]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE id = ?", (doc_id,)
            ).fetchone()
        return _row_to_document(row) if row else None

    def list_documents(self) -> List[DocumentRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM documents ORDER BY created_at DESC"
            ).fetchall()
        return [_row_to_document(r) for r in rows]

    def mark_processed(self, doc_id: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE documents SET processed = 1 WHERE id = ?", (doc_id,)
            )

    # ------------------------------------------------------------------
    # Extractions
    # ------------------------------------------------------------------

    def save_extraction(self, record: ExtractionRecord) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO extractions
                    (document_id, model_used, diagnosis, medications, allergies,
                     laboratory_results, timeline, raw_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.document_id,
                    record.model_used,
                    json.dumps(record.diagnosis),
                    json.dumps(record.medications),
                    json.dumps(record.allergies),
                    json.dumps(record.laboratory_results),
                    json.dumps(record.timeline),
                    record.raw_json,
                    _dt(record.created_at),
                ),
            )
            return cursor.lastrowid  # type: ignore[return-value]

    def get_extractions_for_document(self, doc_id: int) -> List[ExtractionRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM extractions WHERE document_id = ? ORDER BY created_at DESC",
                (doc_id,),
            ).fetchall()
        return [_row_to_extraction(r) for r in rows]

    # ------------------------------------------------------------------
    # Transcriptions
    # ------------------------------------------------------------------

    def save_transcription(self, record: TranscriptionRecord) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO transcriptions
                    (document_id, audio_path, transcript, language, model_used, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.document_id,
                    record.audio_path,
                    record.transcript,
                    record.language,
                    record.model_used,
                    _dt(record.created_at),
                ),
            )
            return cursor.lastrowid  # type: ignore[return-value]

    def get_transcription(self, transcription_id: int) -> Optional[TranscriptionRecord]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM transcriptions WHERE id = ?", (transcription_id,)
            ).fetchone()
        return _row_to_transcription(row) if row else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dt(value: datetime) -> str:
    return value.isoformat()


def _row_to_document(row: sqlite3.Row) -> DocumentRecord:
    return DocumentRecord(
        id=row["id"],
        file_path=row["file_path"],
        file_type=row["file_type"],
        ocr_text=row["ocr_text"],
        created_at=datetime.fromisoformat(row["created_at"]),
        processed=bool(row["processed"]),
    )


def _row_to_extraction(row: sqlite3.Row) -> ExtractionRecord:
    return ExtractionRecord(
        id=row["id"],
        document_id=row["document_id"],
        model_used=row["model_used"],
        diagnosis=json.loads(row["diagnosis"]),
        medications=json.loads(row["medications"]),
        allergies=json.loads(row["allergies"]),
        laboratory_results=json.loads(row["laboratory_results"]),
        timeline=json.loads(row["timeline"]),
        raw_json=row["raw_json"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def _row_to_transcription(row: sqlite3.Row) -> TranscriptionRecord:
    return TranscriptionRecord(
        id=row["id"],
        document_id=row["document_id"],
        audio_path=row["audio_path"],
        transcript=row["transcript"],
        language=row["language"],
        model_used=row["model_used"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )
