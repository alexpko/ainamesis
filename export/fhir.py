"""
fhir.py
Export a MedicalExtraction as a minimal FHIR R4-compatible JSON Bundle.
All resources are generated as in-memory dicts — no external FHIR SDK required.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def export_to_fhir(extraction, output_path: str | Path | None = None) -> str:
    """
    Build a FHIR R4 Bundle (type=collection) from a ``MedicalExtraction``
    and write it to a JSON file.

    Parameters
    ----------
    extraction:
        ``MedicalExtraction`` instance from the LLM module.
    output_path:
        Destination path.  Defaults to ``exports/fhir_<timestamp>.json``.

    Returns
    -------
    str
        Absolute path to the written JSON file.
    """
    if output_path is None:
        exports_dir = Path.cwd() / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = exports_dir / f"fhir_{timestamp}.json"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = _build_bundle(extraction)
    output_path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("FHIR bundle written: %s (%d entries)", output_path, len(bundle["entry"]))
    return str(output_path)


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------

def _build_bundle(extraction) -> Dict[str, Any]:
    entries: List[Dict[str, Any]] = []
    now = datetime.utcnow().isoformat() + "Z"

    # Conditions (diagnosis)
    for dx in extraction.diagnosis:
        entries.append(_wrap_entry(_condition(dx, now)))

    # MedicationStatements
    for med in extraction.medications:
        entries.append(_wrap_entry(_medication_statement(med, now)))

    # AllergyIntolerance
    for allergy in extraction.allergies:
        entries.append(_wrap_entry(_allergy_intolerance(allergy, now)))

    # Observations (lab results)
    for lab in extraction.laboratory_results:
        entries.append(_wrap_entry(_observation(lab, now)))

    return {
        "resourceType": "Bundle",
        "id": str(uuid.uuid4()),
        "type": "collection",
        "timestamp": now,
        "entry": entries,
    }


def _wrap_entry(resource: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "fullUrl": f"urn:uuid:{resource['id']}",
        "resource": resource,
    }


def _condition(diagnosis_text: str, now: str) -> Dict[str, Any]:
    return {
        "resourceType": "Condition",
        "id": str(uuid.uuid4()),
        "recordedDate": now,
        "code": {
            "text": diagnosis_text,
        },
        "clinicalStatus": {
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                "code": "active",
            }]
        },
    }


def _medication_statement(med, now: str) -> Dict[str, Any]:
    resource: Dict[str, Any] = {
        "resourceType": "MedicationStatement",
        "id": str(uuid.uuid4()),
        "status": "active",
        "dateAsserted": now,
        "medication": {"concept": {"text": med.name}},
    }
    dosage: Dict[str, Any] = {}
    if med.route:
        dosage["route"] = {"text": med.route}
    if med.frequency:
        dosage["timing"] = {"code": {"text": med.frequency}}
    if med.dose:
        dosage["doseAndRate"] = [{"doseQuantity": {"value": med.dose}}]
    if dosage:
        resource["dosage"] = [dosage]
    return resource


def _allergy_intolerance(allergy_text: str, now: str) -> Dict[str, Any]:
    return {
        "resourceType": "AllergyIntolerance",
        "id": str(uuid.uuid4()),
        "recordedDate": now,
        "code": {"text": allergy_text},
        "clinicalStatus": {
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/allergyintolerance-clinical",
                "code": "active",
            }]
        },
    }


def _observation(lab, now: str) -> Dict[str, Any]:
    obs: Dict[str, Any] = {
        "resourceType": "Observation",
        "id": str(uuid.uuid4()),
        "status": "final",
        "effectiveDateTime": lab.date or now,
        "code": {"text": lab.test},
    }
    if lab.value is not None:
        try:
            obs["valueQuantity"] = {
                "value": float(lab.value),
                "unit": lab.unit or "",
            }
        except (ValueError, TypeError):
            obs["valueString"] = lab.value
    if lab.reference_range:
        obs["referenceRange"] = [{"text": lab.reference_range}]
    return obs
