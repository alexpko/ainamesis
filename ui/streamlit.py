"""
streamlit.py
Streamlit-based web UI for aInamnesis.
Run with: streamlit run ui/streamlit.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# Allow imports from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from ocr.ocr import OCREngine
from llm.ollama import OllamaLLMClient, MedicalExtraction
from speech.whisper import WhisperTranscriber
from timeline.timeline import TimelineBuilder
from database.models import DocumentRecord, ExtractionRecord
from database.sqlite import AInamesisDB
from export.fhir import export_to_fhir
from export.apple_health import export_to_apple_health

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="aInamnesis",
    page_icon="🩺",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------
def _get_db() -> AInamesisDB:
    if "db" not in st.session_state:
        st.session_state["db"] = AInamesisDB()
    return st.session_state["db"]


def _get_ocr_engine() -> OCREngine:
    if "ocr_engine" not in st.session_state:
        langs = st.session_state.get("ocr_langs", ["en"])
        st.session_state["ocr_engine"] = OCREngine(languages=langs, gpu=False)
    return st.session_state["ocr_engine"]


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🩺 aInamnesis — Offline Medical Document Processor")
st.caption("All processing is 100 % local. No data leaves your device.")

# Sidebar — configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    ollama_model = st.selectbox(
        "Ollama Model",
        ["llama3", "mistral", "medllama2", "llama3:8b"],
        index=0,
    )
    ollama_url = st.text_input("Ollama URL", value="http://localhost:11434")
    whisper_model = st.selectbox("Whisper Model", ["tiny", "base", "small", "medium", "large"], index=2)
    ocr_lang_input = st.text_input("OCR Languages (comma-separated)", value="en")
    st.markdown("---")
    st.markdown("**Local Ollama status**")
    if st.button("Check Ollama"):
        client = OllamaLLMClient(model=ollama_model, base_url=ollama_url)
        if client.is_available():
            st.success("Ollama is running ✓")
        else:
            st.error("Ollama not reachable ✗")

# Main tabs
tab_doc, tab_audio, tab_history, tab_export = st.tabs(
    ["📄 Document", "🎤 Audio", "📋 History", "📤 Export"]
)

# ---------------------------------------------------------------------------
# Tab 1 — Document OCR + LLM extraction
# ---------------------------------------------------------------------------
with tab_doc:
    st.subheader("Upload a medical document")
    uploaded_file = st.file_uploader(
        "Drop a PDF, JPG, or PNG here",
        type=["pdf", "jpg", "jpeg", "png"],
    )

    if uploaded_file and st.button("🔍 Process Document"):
        with st.spinner("Running OCR…"):
            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            ocr_langs = [l.strip() for l in ocr_lang_input.split(",") if l.strip()]
            engine = OCREngine(languages=ocr_langs or ["en"], gpu=False)
            ocr_result = engine.process_file(tmp_path)

        st.success(f"OCR complete — {len(ocr_result.pages)} page(s)")

        with st.expander("Raw OCR text", expanded=False):
            st.text(ocr_result.full_text[:3000] + ("…" if len(ocr_result.full_text) > 3000 else ""))

        with st.spinner("Extracting clinical data via LLM…"):
            llm_client = OllamaLLMClient(model=ollama_model, base_url=ollama_url)
            extraction = llm_client.extract(ocr_result.full_text)

        st.session_state["last_extraction"] = extraction

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🩺 Diagnosis")
            for dx in extraction.diagnosis or ["—"]:
                st.markdown(f"- {dx}")

            st.markdown("### 💊 Medications")
            if extraction.medications:
                for med in extraction.medications:
                    st.markdown(f"- **{med.name}** {med.dose or ''} {med.frequency or ''}")
            else:
                st.markdown("—")

        with col2:
            st.markdown("### ⚠️ Allergies")
            for a in extraction.allergies or ["—"]:
                st.markdown(f"- {a}")

            st.markdown("### 🧪 Laboratory Results")
            if extraction.laboratory_results:
                for lab in extraction.laboratory_results:
                    st.markdown(f"- **{lab.test}**: {lab.value or '?'} {lab.unit or ''}")
            else:
                st.markdown("—")

        st.markdown("### 📅 Timeline")
        timeline = TimelineBuilder().build(extraction)
        for entry in timeline.sorted_entries():
            st.markdown(f"- **{entry.display_date()}** — {entry.event}")

        # Persist to DB
        db = _get_db()
        doc_id = db.save_document(DocumentRecord(
            file_path=uploaded_file.name,
            file_type=suffix.lstrip("."),
            ocr_text=ocr_result.full_text,
            processed=True,
        ))
        db.save_extraction(ExtractionRecord(
            document_id=doc_id,
            model_used=ollama_model,
            diagnosis=extraction.diagnosis,
            medications=[m.model_dump() for m in extraction.medications],
            allergies=extraction.allergies,
            laboratory_results=[l.model_dump() for l in extraction.laboratory_results],
            timeline=[t.model_dump() for t in extraction.timeline],
        ))
        st.info(f"Saved to database (document_id={doc_id})")

# ---------------------------------------------------------------------------
# Tab 2 — Audio transcription
# ---------------------------------------------------------------------------
with tab_audio:
    st.subheader("Upload an audio consultation")
    audio_file = st.file_uploader(
        "Drop an audio file (mp3, wav, m4a, flac…)",
        type=["mp3", "wav", "m4a", "flac", "ogg"],
        key="audio_uploader",
    )

    if audio_file and st.button("🎙️ Transcribe"):
        with st.spinner(f"Transcribing with Whisper '{whisper_model}'…"):
            suffix = Path(audio_file.name).suffix
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(audio_file.read())
                tmp_path = tmp.name

            transcriber = WhisperTranscriber(model_name=whisper_model)
            transcript = transcriber.transcribe_file(tmp_path)

        st.text_area("Transcript", transcript, height=200)

        if st.button("Extract clinical data from transcript"):
            llm_client = OllamaLLMClient(model=ollama_model, base_url=ollama_url)
            extraction = llm_client.extract(transcript)
            st.session_state["last_extraction"] = extraction
            st.json(extraction.model_dump())

# ---------------------------------------------------------------------------
# Tab 3 — History
# ---------------------------------------------------------------------------
with tab_history:
    st.subheader("Processed documents")
    db = _get_db()
    docs = db.list_documents()
    if docs:
        for doc in docs:
            with st.expander(f"[{doc.id}] {Path(doc.file_path).name} — {doc.created_at.date()}"):
                st.text(doc.ocr_text[:500] + "…" if doc.ocr_text and len(doc.ocr_text) > 500 else doc.ocr_text or "")
                extractions = db.get_extractions_for_document(doc.id)  # type: ignore
                for ext in extractions:
                    st.markdown(f"**Extraction** (model={ext.model_used})")
                    st.json({"diagnosis": ext.diagnosis, "allergies": ext.allergies})
    else:
        st.info("No documents processed yet.")

# ---------------------------------------------------------------------------
# Tab 4 — Export
# ---------------------------------------------------------------------------
with tab_export:
    st.subheader("Export last extraction")
    extraction: MedicalExtraction | None = st.session_state.get("last_extraction")

    if extraction is None:
        st.warning("Process a document first.")
    else:
        col_fhir, col_ah = st.columns(2)
        with col_fhir:
            if st.button("Export as FHIR R4 JSON"):
                path = export_to_fhir(extraction)
                st.success(f"Written: {path}")
                with open(path) as f:
                    st.download_button("Download FHIR JSON", f.read(), file_name="fhir_export.json")

        with col_ah:
            if st.button("Export as Apple Health XML"):
                path = export_to_apple_health(extraction)
                st.success(f"Written: {path}")
                with open(path) as f:
                    st.download_button("Download Apple Health XML", f.read(), file_name="apple_health_export.xml")
