"""
prompts.py
Prompt templates for the LLM extraction pipeline.
"""
from __future__ import annotations

SYSTEM_PROMPT = """
You are a medical data extraction assistant.
Extract data from the text and return ONLY a valid JSON object.
Do not write conversational text.

Required JSON format:
{
    "diagnosis": "Extract diagnosis or write '-'",
    "allergies": "Extract allergies or write '-'",
    "medications": "Extract medications or write '-'",
    "laboratory_results": "Extract lab results or write '-'",
    "timeline": "Extract dates and upcoming appointments or write '-'"
}

Example output:
{
    "diagnosis": "Abrupce mediálního kondylu femuru",
    "allergies": "-",
    "medications": "Ibalgin 400mg",
    "laboratory_results": "-",
    "timeline": "Kontrola 12.8.2026"
}

Correct OCR errors based on medical context.
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
