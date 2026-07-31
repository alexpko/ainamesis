"""
test_llm.py
Unit tests for the LLM module (ollama.py + prompts.py).
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

class TestPrompts:
    def test_extraction_prompt_contains_schema_keys(self):
        from llm.prompts import build_extraction_prompt
        prompt = build_extraction_prompt("Patient: John, 45. Hypertension.")
        for key in ("diagnosis", "medications", "allergies", "laboratory_results", "timeline"):
            assert key in prompt

    def test_extraction_prompt_contains_ocr_text(self):
        from llm.prompts import build_extraction_prompt
        ocr = "Blood pressure: 140/90"
        prompt = build_extraction_prompt(ocr)
        assert ocr in prompt

    def test_summary_prompt_contains_json(self):
        from llm.prompts import build_summary_prompt
        json_data = '{"diagnosis": ["Hypertension"]}'
        prompt = build_summary_prompt(json_data)
        assert json_data in prompt


# ---------------------------------------------------------------------------
# MedicalExtraction model
# ---------------------------------------------------------------------------

class TestMedicalExtraction:
    def test_defaults_are_empty_lists(self):
        from llm.ollama import MedicalExtraction
        m = MedicalExtraction()
        assert m.diagnosis == []
        assert m.medications == []
        assert m.allergies == []
        assert m.laboratory_results == []
        assert m.timeline == []

    def test_full_construction(self):
        from llm.ollama import MedicalExtraction, MedicationEntry, LabResult, TimelineEvent
        m = MedicalExtraction(
            diagnosis=["Hypertension"],
            medications=[MedicationEntry(name="Amlodipine", dose="5mg", frequency="once daily")],
            allergies=["Penicillin"],
            laboratory_results=[LabResult(test="HbA1c", value="6.1", unit="%")],
            timeline=[TimelineEvent(date="2024-01-15", event="Diagnosis confirmed")],
        )
        assert m.diagnosis == ["Hypertension"]
        assert m.medications[0].name == "Amlodipine"
        assert m.allergies[0] == "Penicillin"
        assert m.laboratory_results[0].test == "HbA1c"
        assert m.timeline[0].event == "Diagnosis confirmed"


# ---------------------------------------------------------------------------
# OllamaLLMClient
# ---------------------------------------------------------------------------

SAMPLE_JSON = {
    "diagnosis": ["Type 2 Diabetes"],
    "medications": [{"name": "Metformin", "dose": "500mg", "frequency": "twice daily", "route": "oral"}],
    "allergies": ["Sulfa drugs"],
    "laboratory_results": [{"test": "Fasting glucose", "value": "126", "unit": "mg/dL", "reference_range": "70-100", "date": "2024-03-01"}],
    "timeline": [{"date": "2024-03-01", "event": "Fasting glucose elevated"}],
}


class TestOllamaLLMClient:
    def _make_client(self, model: str = "llama3") -> "OllamaLLMClient":
        from llm.ollama import OllamaLLMClient
        return OllamaLLMClient(model=model, base_url="http://localhost:11434", timeout=10)

    def test_extract_returns_empty_on_empty_input(self):
        client = self._make_client()
        result = client.extract("")
        assert result.diagnosis == []

    def test_extract_parses_valid_json_response(self):
        client = self._make_client()
        raw_response = json.dumps(SAMPLE_JSON)
        with patch.object(client, "_call_ollama", return_value=raw_response):
            result = client.extract("Patient has diabetes.")
        assert result.diagnosis == ["Type 2 Diabetes"]
        assert result.medications[0].name == "Metformin"
        assert result.allergies[0] == "Sulfa drugs"
        assert result.laboratory_results[0].test == "Fasting glucose"
        assert result.timeline[0].date == "2024-03-01"

    def test_extract_handles_fenced_json(self):
        client = self._make_client()
        raw_response = "```json\n" + json.dumps(SAMPLE_JSON) + "\n```"
        with patch.object(client, "_call_ollama", return_value=raw_response):
            result = client.extract("Patient has diabetes.")
        assert result.diagnosis == ["Type 2 Diabetes"]

    def test_extract_returns_empty_on_connection_error(self):
        client = self._make_client()
        with patch.object(client, "_call_ollama", return_value=None):
            result = client.extract("Some text.")
        assert result.diagnosis == []

    def test_extract_returns_empty_on_malformed_json(self):
        client = self._make_client()
        with patch.object(client, "_call_ollama", return_value="not json at all {{}"):
            result = client.extract("Some text.")
        assert isinstance(result.diagnosis, list)

    def test_is_available_true(self):
        client = self._make_client()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch.object(client._requests, "get", return_value=mock_resp):
            assert client.is_available() is True

    def test_is_available_false_on_connection_error(self):
        import requests
        client = self._make_client()
        with patch.object(client._requests, "get", side_effect=requests.exceptions.ConnectionError):
            assert client.is_available() is False

    def test_extract_json_block_bare(self):
        from llm.ollama import OllamaLLMClient
        result = OllamaLLMClient._extract_json_block('{"key": "val"}')
        assert result == '{"key": "val"}'

    def test_extract_json_block_fenced(self):
        from llm.ollama import OllamaLLMClient
        result = OllamaLLMClient._extract_json_block("```json\n{\"k\": 1}\n```")
        assert result == '{"k": 1}'

    def test_extract_json_block_none(self):
        from llm.ollama import OllamaLLMClient
        result = OllamaLLMClient._extract_json_block("no json here")
        assert result is None
