"""
prompts.py
Prompt templates for the LLM extraction pipeline.
"""
from __future__ import annotations

_SYSTEM_CONTEXT = """\
You are a medical AI assistant. Your sole task is to extract structured \
clinical information from raw medical document text provided by the user. \
You must return ONLY valid JSON — no prose, no markdown fences, no explanations.\
"""

_EXTRACTION_SCHEMA = """\
{
  "diagnosis": ["string", ...],
  "medications": [
    {
      "name": "string",
      "dose": "string or null",
      "frequency": "string or null",
      "route": "string or null"
    }
  ],
  "allergies": ["string", ...],
  "laboratory_results": [
    {
      "test": "string",
      "value": "string or null",
      "unit": "string or null",
      "reference_range": "string or null",
      "date": "string or null"
    }
  ],
  "timeline": [
    {
      "date": "string or null",
      "event": "string"
    }
  ]
}\
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
        f"{_SYSTEM_CONTEXT}\n\n"
        "Extract ALL relevant clinical data from the following medical document text.\n"
        "Return your answer as a single JSON object that conforms EXACTLY to this schema:\n\n"
        f"{_EXTRACTION_SCHEMA}\n\n"
        "Rules:\n"
        "- Use null for missing optional fields, never omit them.\n"
        "- Dates must be in ISO-8601 format (YYYY-MM-DD) when determinable, otherwise use the "
        "original string.\n"
        "- Do NOT hallucinate data that is not present in the source text.\n"
        "- If a category has no findings, return an empty array [].\n\n"
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
        f"{_SYSTEM_CONTEXT}\n\n"
        "Given the following structured medical data (JSON), write a concise clinical "
        "summary in plain English suitable for a physician's note. "
        "Return ONLY the summary text, no JSON.\n\n"
        f"{extraction_json}"
    )
