"""
main.py
aInamnesis CLI entry-point.

Usage
-----
    # Process a single document
    python app/main.py --file path/to/scan.pdf

    # Process + transcribe audio
    python app/main.py --audio path/to/consultation.mp3

    # Override Ollama model
    python app/main.py --file scan.pdf --model mistral

    # Export to FHIR after processing
    python app/main.py --file scan.pdf --export fhir

    # Start Streamlit UI
    python app/main.py --ui
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr.ocr import OCREngine
from llm.ollama import OllamaLLMClient
from speech.whisper import WhisperTranscriber
from timeline.timeline import TimelineBuilder
from database.models import DocumentRecord, ExtractionRecord, TranscriptionRecord
from database.sqlite import AInamesisDB

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def process_document(
    file_path: str,
    model: str,
    ollama_url: str,
    languages: list[str],
    export_format: str | None,
) -> None:
    """Full OCR → LLM → DB pipeline for a single document."""
    path = Path(file_path)
    if not path.exists():
        logger.error("File not found: %s", path)
        sys.exit(1)

    # --- OCR ---
    logger.info("Running OCR on: %s", path.name)
    engine = OCREngine(languages=languages, gpu=False)
    ocr_result = engine.process_file(path)
    logger.info("OCR complete. Characters extracted: %d", len(ocr_result.full_text))

    # --- LLM ---
    client = OllamaLLMClient(model=model, base_url=ollama_url)
    if not client.is_available():
        logger.error("Ollama is not running at %s. Start it with: ollama serve", ollama_url)
        sys.exit(1)

    logger.info("Extracting clinical data (model=%s)…", model)
    extraction = client.extract(ocr_result.full_text)

    # --- Timeline ---
    timeline = TimelineBuilder().build(extraction)

    # --- Display ---
    print("\n" + "=" * 60)
    print(f"  aInamnesis — Extraction Report")
    print("=" * 60)

    def section(title: str, items: list) -> None:
        print(f"\n  {title}")
        print("  " + "-" * (len(title)))
        for item in items:
            print(f"    • {item}")
        if not items:
            print("    (none)")

    section("Diagnosis", extraction.diagnosis)
    section("Allergies", extraction.allergies)
    section(
        "Medications",
        [f"{m.name} {m.dose or ''} {m.frequency or ''}".strip() for m in extraction.medications],
    )
    section(
        "Laboratory Results",
        [f"{l.test}: {l.value or '?'} {l.unit or ''}" for l in extraction.laboratory_results],
    )
    section(
        "Timeline",
        [f"{e.display_date()} — {e.event}" for e in timeline.sorted_entries()],
    )
    print("=" * 60 + "\n")

    # --- Persist ---
    db = AInamesisDB()
    doc_id = db.save_document(DocumentRecord(
        file_path=str(path),
        file_type=path.suffix.lstrip("."),
        ocr_text=ocr_result.full_text,
        processed=True,
    ))
    db.save_extraction(ExtractionRecord(
        document_id=doc_id,
        model_used=model,
        diagnosis=extraction.diagnosis,
        medications=[m.model_dump() for m in extraction.medications],
        allergies=extraction.allergies,
        laboratory_results=[l.model_dump() for l in extraction.laboratory_results],
        timeline=[t.model_dump() for t in extraction.timeline],
    ))
    logger.info("Saved to database (document_id=%d)", doc_id)

    # --- Export ---
    if export_format == "fhir":
        from export.fhir import export_to_fhir
        out = export_to_fhir(extraction)
        logger.info("FHIR export: %s", out)
    elif export_format == "apple_health":
        from export.apple_health import export_to_apple_health
        out = export_to_apple_health(extraction)
        logger.info("Apple Health export: %s", out)
    elif export_format == "pdf":
        from export.pdf import export_to_pdf
        out = export_to_pdf(extraction)
        logger.info("PDF report: %s", out)


def process_audio(
    audio_path: str,
    model: str,
    ollama_url: str,
    whisper_model: str,
) -> None:
    """Whisper → LLM pipeline for an audio consultation."""
    path = Path(audio_path)
    if not path.exists():
        logger.error("Audio file not found: %s", path)
        sys.exit(1)

    logger.info("Transcribing: %s (whisper=%s)", path.name, whisper_model)
    transcriber = WhisperTranscriber(model_name=whisper_model)
    transcript = transcriber.transcribe_file(path)
    logger.info("Transcript (%d chars):\n%s", len(transcript), transcript[:300])

    client = OllamaLLMClient(model=model, base_url=ollama_url)
    logger.info("Extracting clinical data from transcript…")
    extraction = client.extract(transcript)

    db = AInamesisDB()
    transcription_id = db.save_transcription(TranscriptionRecord(
        audio_path=str(path),
        transcript=transcript,
        model_used=whisper_model,
    ))
    db.save_extraction(ExtractionRecord(
        document_id=transcription_id,
        model_used=model,
        diagnosis=extraction.diagnosis,
        medications=[m.model_dump() for m in extraction.medications],
        allergies=extraction.allergies,
        laboratory_results=[l.model_dump() for l in extraction.laboratory_results],
        timeline=[t.model_dump() for t in extraction.timeline],
    ))
    print(json.dumps(extraction.model_dump(), indent=2, ensure_ascii=False))


def launch_ui() -> None:
    """Start the Streamlit web interface."""
    ui_path = Path(__file__).resolve().parent.parent / "ui" / "streamlit.py"
    logger.info("Launching Streamlit UI: %s", ui_path)
    subprocess.run(["streamlit", "run", str(ui_path)], check=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ainamesis",
        description="aInamnesis — Offline medical document processor",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", metavar="PATH", help="Path to a PDF / JPG / PNG document")
    group.add_argument("--audio", metavar="PATH", help="Path to an audio consultation file")
    group.add_argument("--ui", action="store_true", help="Launch the Streamlit web UI")

    parser.add_argument("--model", default="llama3", help="Ollama model name (default: llama3)")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--whisper-model", default="small", help="Whisper model size (default: small)")
    parser.add_argument(
        "--langs", default="en",
        help="Comma-separated OCR language codes (default: en)"
    )
    parser.add_argument(
        "--export",
        choices=["fhir", "apple_health", "pdf"],
        default=None,
        help="Export format after processing",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.ui:
        launch_ui()
    elif args.file:
        process_document(
            file_path=args.file,
            model=args.model,
            ollama_url=args.ollama_url,
            languages=[l.strip() for l in args.langs.split(",")],
            export_format=args.export,
        )
    elif args.audio:
        process_audio(
            audio_path=args.audio,
            model=args.model,
            ollama_url=args.ollama_url,
            whisper_model=args.whisper_model,
        )


if __name__ == "__main__":
    main()
