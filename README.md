# aInamnesis — Offline Medical Document Processor

> **Privacy-first AI pipeline for medical documentation.**  
> Every byte stays on your machine — no cloud, no telemetry, no accounts.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Examples](#examples)
5. [Module Reference](#module-reference)
6. [Running Tests](#running-tests)
7. [Export Formats](#export-formats)

---

## Architecture

```
aInamnesis/
│
├── app/
│   └── main.py               CLI entry-point — orchestrates the full pipeline
│
├── ocr/
│   ├── pdf_loader.py         PDF→PIL page converter (via pdf2image / poppler)
│   └── ocr.py                EasyOCR primary engine + Tesseract fallback
│
├── speech/
│   └── whisper.py            Local Whisper transcription (openai-whisper)
│
├── llm/
│   ├── prompts.py            Prompt templates for medical extraction
│   └── ollama.py             Ollama client → structured MedicalExtraction model
│
├── timeline/
│   └── timeline.py           Chronological event extraction and sorting
│
├── database/
│   ├── models.py             Pydantic models (DocumentRecord, ExtractionRecord, …)
│   └── sqlite.py             SQLite repository — WAL mode, thread-safe
│
├── export/
│   ├── pdf.py                ReportLab PDF report
│   ├── fhir.py               FHIR R4 JSON Bundle
│   └── apple_health.py       Apple Health XML export
│
├── ui/
│   └── streamlit.py          Streamlit web UI (drag-and-drop, history, exports)
│
└── tests/
    ├── conftest.py
    ├── test_ocr.py
    ├── test_llm.py
    ├── test_timeline.py
    ├── test_speech.py
    ├── test_database.py
    └── test_export.py
```

### Data Flow

```
PDF / JPG / PNG         Audio file
      │                      │
      ▼                      ▼
  OCREngine            WhisperTranscriber
  (EasyOCR +           (openai-whisper,
   Tesseract)           local model)
      │                      │
      └──────────┬───────────┘
                 ▼
         OllamaLLMClient
         (llama3 / mistral / …)
                 │
                 ▼
        MedicalExtraction
    ┌────────────┼────────────┐
    ▼            ▼            ▼
TimelineBuilder  SQLite DB   Export
                          (PDF / FHIR /
                           Apple Health)
```

---

## Installation

### Prerequisites

| Dependency | Install |
|------------|---------|
| Python 3.10+ | [python.org](https://www.python.org/) |
| Poppler (for PDF support) | macOS: `brew install poppler` · Ubuntu: `apt install poppler-utils` |
| Tesseract OCR | macOS: `brew install tesseract` · Ubuntu: `apt install tesseract-ocr` |
| Ollama | [ollama.com](https://ollama.com) — `ollama pull llama3` |

### Python packages

```bash
# Clone the repo
git clone https://github.com/your-org/ainamesis.git
cd ainamesis

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Pull an Ollama model

```bash
ollama pull llama3          # recommended (8B)
# or
ollama pull mistral
# or — for dedicated medical fine-tunes —
ollama pull medllama2
```

---

## Usage

### CLI

```bash
# Process a PDF document
python app/main.py --file path/to/scan.pdf

# Multi-language OCR (English + German)
python app/main.py --file scan.pdf --langs en,de

# Use a different Ollama model
python app/main.py --file scan.pdf --model mistral

# Process + export to FHIR
python app/main.py --file scan.pdf --export fhir

# Transcribe an audio consultation
python app/main.py --audio consultation.mp3 --whisper-model medium

# Launch the Streamlit web UI
python app/main.py --ui
# — or directly —
streamlit run ui/streamlit.py
```

### Web UI

After running `streamlit run ui/streamlit.py`, open **http://localhost:8501** in your browser.

Tabs:
- **📄 Document** — drag-and-drop PDF/JPG/PNG → OCR → LLM extraction
- **🎤 Audio** — upload audio → Whisper transcription → optional LLM extraction
- **📋 History** — browse all previously processed documents
- **📤 Export** — download last extraction as FHIR JSON or Apple Health XML

---

## Examples

### Python API

```python
from ocr.ocr import OCREngine
from llm.ollama import OllamaLLMClient
from timeline.timeline import TimelineBuilder
from export.fhir import export_to_fhir

# 1. OCR
engine = OCREngine(languages=["en", "de"], gpu=False)
ocr_result = engine.process_file("lab_report.pdf")
print(ocr_result.full_text)

# 2. LLM extraction
client = OllamaLLMClient(model="llama3")
extraction = client.extract(ocr_result.full_text)

print(extraction.diagnosis)         # ['Type 2 Diabetes Mellitus']
print(extraction.allergies)         # ['Penicillin']
print(extraction.medications[0])    # MedicationEntry(name='Metformin', dose='500mg', ...)

# 3. Timeline
timeline = TimelineBuilder().build(extraction)
for event in timeline.sorted_entries():
    print(event.display_date(), "—", event.event)

# 4. Export
fhir_path = export_to_fhir(extraction)
print("FHIR bundle written to:", fhir_path)
```

### Speech-to-structured notes

```python
from speech.whisper import WhisperTranscriber
from llm.ollama import OllamaLLMClient

transcriber = WhisperTranscriber(model_name="small", language="en")
transcript = transcriber.transcribe_file("consultation_20240315.mp3")

client = OllamaLLMClient(model="llama3")
extraction = client.extract(transcript)

import json
print(json.dumps(extraction.model_dump(), indent=2))
```

### Persist to SQLite

```python
from database.sqlite import AInamesisDB
from database.models import DocumentRecord, ExtractionRecord

db = AInamesisDB()   # creates data/ainamesis.db automatically

doc_id = db.save_document(DocumentRecord(
    file_path="scan.pdf",
    file_type="pdf",
    ocr_text=ocr_result.full_text,
    processed=True,
))

db.save_extraction(ExtractionRecord(
    document_id=doc_id,
    model_used="llama3",
    diagnosis=extraction.diagnosis,
    medications=[m.model_dump() for m in extraction.medications],
    allergies=extraction.allergies,
    laboratory_results=[l.model_dump() for l in extraction.laboratory_results],
    timeline=[t.model_dump() for t in extraction.timeline],
))
```

---

## Module Reference

| Module | Class / Function | Description |
|--------|-----------------|-------------|
| `ocr.pdf_loader` | `load_document()` | Generator yielding PIL images per page |
| `ocr.ocr` | `OCREngine` | EasyOCR + Tesseract, multi-language |
| `llm.prompts` | `build_extraction_prompt()` | Structured medical extraction prompt |
| `llm.ollama` | `OllamaLLMClient` | Calls local Ollama, returns `MedicalExtraction` |
| `llm.ollama` | `MedicalExtraction` | Pydantic model (diagnosis, medications, allergies, lab results, timeline) |
| `speech.whisper` | `WhisperTranscriber` | Local audio transcription |
| `timeline.timeline` | `TimelineBuilder` | Builds sorted `MedicalTimeline` |
| `database.sqlite` | `AInamesisDB` | SQLite CRUD repository |
| `database.models` | `DocumentRecord` etc. | Pydantic DB models |
| `export.fhir` | `export_to_fhir()` | FHIR R4 JSON Bundle |
| `export.apple_health` | `export_to_apple_health()` | Apple Health XML |
| `export.pdf` | `export_to_pdf()` | PDF report (ReportLab) |
| `ui.streamlit` | — | Streamlit web UI |

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=. --cov-report=term-missing

# Single module
pytest tests/test_ocr.py -v
```

Test files cover:
- `test_ocr.py` — pdf_loader, OCREngine (EasyOCR + Tesseract paths)
- `test_llm.py` — prompts, MedicalExtraction model, OllamaLLMClient (mocked HTTP)
- `test_timeline.py` — date parsing, sorting, deduplication, lab enrichment
- `test_speech.py` — WhisperTranscriber (mocked model), error handling
- `test_database.py` — full SQLite CRUD for all record types
- `test_export.py` — FHIR bundle structure, Apple Health XML, empty extractions

---

## Export Formats

### FHIR R4 JSON
Produces a `Bundle` (type=`collection`) with `Condition`, `MedicationStatement`, `AllergyIntolerance`, and `Observation` resources. Compatible with any FHIR R4 server.

### Apple Health XML
Follows the Apple Health Export schema with `ClinicalRecord` elements typed as `HKClinicalTypeIdentifier*`. Can be imported via third-party tools (e.g. Health Auto Export).

### PDF Report
A formatted A4 PDF with sections for Diagnosis, Allergies, Medications (table), Laboratory Results (table), and Timeline. Generated with ReportLab — no cloud service.

---

## Privacy & Security

- **Zero network requests** during processing (except to your local Ollama instance)
- All data stored in a local SQLite file (`data/ainamesis.db`)
- No telemetry, analytics, or external API calls
- OCR logs written locally to `logs/ocr/`

---

*Built with ❤️ for privacy-first medical AI — powered by IBM Bob.*
