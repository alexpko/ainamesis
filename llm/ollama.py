"""
ollama.py
Local LLM inference via Ollama.
Parses OCR text into structured medical data:
  diagnosis, medications, allergies, laboratory_results, timeline
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class MedicationEntry(BaseModel):
    name: str
    dose: Optional[str] = None
    frequency: Optional[str] = None
    route: Optional[str] = None


class LabResult(BaseModel):
    test: str
    value: Optional[str] = None
    unit: Optional[str] = None
    reference_range: Optional[str] = None
    date: Optional[str] = None


class TimelineEvent(BaseModel):
    date: Optional[str] = None
    event: str


class MedicalExtraction(BaseModel):
    """Structured output produced by the LLM module."""
    diagnosis: List[str] = Field(default_factory=list)
    medications: List[MedicationEntry] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    laboratory_results: List[LabResult] = Field(default_factory=list)
    timeline: List[TimelineEvent] = Field(default_factory=list)
    raw_llm_response: str = Field(default="", exclude=True)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class OllamaLLMClient:
    """
    Sends OCR text to a locally running Ollama instance and returns a
    validated ``MedicalExtraction`` object.

    Parameters
    ----------
    model:
        Ollama model tag, e.g. "llama3", "mistral", "medllama2".
    base_url:
        Base URL of the local Ollama API.  Defaults to http://localhost:11434.
    timeout:
        Request timeout in seconds.
    """

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._api_url = f"{self.base_url}/api/generate"

        # Lazy import — requests is a lightweight dep already in requirements
        import requests  # noqa: PLC0415
        self._requests = requests

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, ocr_text: str) -> MedicalExtraction:
        """
        Extract structured medical information from *ocr_text*.

        Returns
        -------
        MedicalExtraction
            Validated Pydantic model.  Falls back to empty model on error.
        """
        if not ocr_text.strip():
            logger.warning("Empty OCR text received. Returning empty extraction.")
            return MedicalExtraction()

        from llm.prompts import build_extraction_prompt  # noqa: PLC0415
        prompt = build_extraction_prompt(ocr_text)

        logger.info("Sending request to Ollama (model=%s, chars=%d)…", self.model, len(ocr_text))
        raw_response = self._call_ollama(prompt)

        if raw_response is None:
            return MedicalExtraction()

        extraction = self._parse_response(raw_response)
        extraction.raw_llm_response = raw_response
        return extraction

    def is_available(self) -> bool:
        """Return True when the Ollama server is reachable."""
        try:
            resp = self._requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_ollama(self, prompt: str) -> Optional[str]:
        """POST to Ollama generate endpoint and return the response text."""
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,   # low temperature for factual extraction
                "num_predict": 2048,
            },
        }
        try:
            response = self._requests.post(
                self._api_url,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
        except self._requests.exceptions.ConnectionError:
            logger.error(
                "Cannot connect to Ollama at %s. Is it running?", self.base_url
            )
        except self._requests.exceptions.Timeout:
            logger.error("Ollama request timed out after %ds.", self.timeout)
        except self._requests.exceptions.HTTPError as exc:
            logger.error("Ollama HTTP error: %s", exc)
        except (KeyError, ValueError) as exc:
            logger.error("Unexpected Ollama response format: %s", exc)
        return None

    def _parse_response(self, raw: str) -> MedicalExtraction:
        """Extract JSON from the LLM response and validate it."""
        # Step 1: pull JSON text out of whatever the model wrapped it in
        json_str = self._extract_json_block(raw)
        if not json_str:
            logger.warning("No JSON block found in LLM response. Attempting full parse.")
            json_str = raw.strip()

        # Step 2: parse to dict
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            logger.error("JSON decode error: %s\n--- raw ---\n%s", exc, raw[:500])
            return MedicalExtraction()

        # Step 3: normalise flat strings → list/model shapes expected by Pydantic.
        # The current SYSTEM_PROMPT returns plain strings per field; convert them
        # so the rest of the pipeline (timeline builder, exports, DB) works unchanged.
        data = self._normalise_data(data)

        # Step 4: validate with Pydantic
        try:
            return MedicalExtraction(**data)
        except Exception as exc:
            logger.error("Pydantic validation error: %s", exc)
        return MedicalExtraction()

    @staticmethod
    def _extract_json_block(text: str) -> Optional[str]:
        """Pull the first ```json … ``` fence or bare { … } block from text.

        Uses greedy ``.*`` (not ``.*?``) so multi-line JSON inside a fence is
        captured in full.
        """
        # Fenced code block: ```json ... ``` or ``` ... ```
        fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
        if fence_match:
            return fence_match.group(1)
        # Bare JSON object — greedily match outermost { … }
        brace_match = re.search(r"(\{.*\})", text, re.DOTALL)
        if brace_match:
            return brace_match.group(1)
        return None

    @staticmethod
    def _normalise_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Coerce flat string values returned by the simple SYSTEM_PROMPT into
        the list/nested-model shapes that ``MedicalExtraction`` expects.

        - ``diagnosis``          str  → [str]  (or keep list as-is)
        - ``allergies``          str  → [str]
        - ``medications``        str  → [MedicationEntry-compatible dict]
        - ``laboratory_results`` str  → [LabResult-compatible dict]
        - ``timeline``           str  → [TimelineEvent-compatible dict]
        A value of ``"-"`` is treated as empty.
        """
        def _as_list_of_str(val) -> list:
            if not val or val == "-":
                return []
            if isinstance(val, list):
                return val
            # Split on commas or semicolons for multi-value strings
            return [s.strip() for s in re.split(r"[;,]\s*", str(val)) if s.strip()]

        def _as_medication_list(val) -> list:
            if not val or val == "-":
                return []
            if isinstance(val, list):
                return val
            # Each comma/semicolon-separated token becomes a minimal entry
            return [{"name": s.strip()} for s in re.split(r"[;,]\s*", str(val)) if s.strip()]

        def _as_lab_list(val) -> list:
            if not val or val == "-":
                return []
            if isinstance(val, list):
                return val
            return [{"test": s.strip()} for s in re.split(r"[;,]\s*", str(val)) if s.strip()]

        def _as_timeline_list(val) -> list:
            if not val or val == "-":
                return []
            if isinstance(val, list):
                return val
            return [{"date": None, "event": s.strip()} for s in re.split(r"[;,]\s*", str(val)) if s.strip()]

        return {
            "diagnosis":          _as_list_of_str(data.get("diagnosis")),
            "allergies":          _as_list_of_str(data.get("allergies")),
            "medications":        _as_medication_list(data.get("medications")),
            "laboratory_results": _as_lab_list(data.get("laboratory_results")),
            "timeline":           _as_timeline_list(data.get("timeline")),
        }
