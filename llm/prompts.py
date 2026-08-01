"""
prompts.py
Prompt templates for the LLM extraction pipeline.
"""
from __future__ import annotations

SYSTEM_PROMPT = """
You are a professional medical assistant. 
Extract the following data from the provided text strictly in JSON format:
{
    "diagnosis": "brief diagnosis",
    "allergies": "list of allergies or '-'",
    "medications": "list of medications",
    "laboratory_results": "laboratory results",
    "timeline": "key dates and events"
}
If no data is found, write '-'. 
The text contains OCR errors; correct them based on medical context.
"""


def build_extraction_prompt(ocr_text: str) -> str:
    """
    Compose the full prompt sent to Ollama.

    Parameters
    ----------
    ocr_text:
        Raw text extracted from the medical document via OCR.

    Returns
    -------
    str
        Complete prompt string ready for the generate endpoint.
    """
    return (
        f"{SYSTEM_PROMPT}\n"
        "Medical document text:\n"
        "---\n"
        f"{ocr_text}\n"
        "---\n\n"
        "JSON output:"
    )


def build_summary_prompt(extraction_json: str) -> str:
    """
    Build a prompt that asks the LLM to produce a human-readable clinical
    summary from an already-parsed MedicalExtraction JSON.
    """
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "Given the following structured medical data (JSON), write a concise clinical "
        "summary in plain English suitable for a physician's note. "
        "Return ONLY the summary text, no JSON.\n\n"
        f"{extraction_json}"
    )
