"""
apple_health.py
Export a MedicalExtraction as a minimal Apple Health-compatible XML file.
The format follows the Apple Health Export XML schema so it can be imported
back into the Health app via third-party tools (e.g. Health Auto Export).
"""
from __future__ import annotations

import logging
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_DATE_FMT = "%Y-%m-%d %H:%M:%S +0000"


def export_to_apple_health(extraction, output_path: str | Path | None = None) -> str:
    """
    Build an Apple Health XML export from a ``MedicalExtraction``.

    Parameters
    ----------
    extraction:
        ``MedicalExtraction`` instance from the LLM module.
    output_path:
        Destination path.  Defaults to ``exports/apple_health_<timestamp>.xml``.

    Returns
    -------
    str
        Absolute path to the written XML file.
    """
    if output_path is None:
        exports_dir = Path.cwd() / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = exports_dir / f"apple_health_{timestamp}.xml"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    root = ET.Element("HealthData", locale="en_US")
    now = datetime.utcnow().strftime(_DATE_FMT)

    # Clinical Notes — Diagnosis entries as ClinicalRecord
    for dx in extraction.diagnosis:
        record = ET.SubElement(root, "ClinicalRecord")
        record.set("type", "HKClinicalTypeIdentifierConditionRecord")
        record.set("identifier", str(uuid.uuid4()))
        record.set("sourceRevision", "aInamnesis")
        record.set("startDate", now)
        record.set("endDate", now)
        record.set("displayName", dx)
        record.set("fhirVersion", "4.0.1")

    # Medications
    for med in extraction.medications:
        record = ET.SubElement(root, "ClinicalRecord")
        record.set("type", "HKClinicalTypeIdentifierMedicationRecord")
        record.set("identifier", str(uuid.uuid4()))
        record.set("sourceRevision", "aInamnesis")
        record.set("startDate", now)
        record.set("endDate", now)
        record.set("displayName", med.name)
        if med.dose:
            record.set("dose", med.dose)
        if med.frequency:
            record.set("frequency", med.frequency)

    # Allergies
    for allergy in extraction.allergies:
        record = ET.SubElement(root, "ClinicalRecord")
        record.set("type", "HKClinicalTypeIdentifierAllergyRecord")
        record.set("identifier", str(uuid.uuid4()))
        record.set("sourceRevision", "aInamnesis")
        record.set("startDate", now)
        record.set("endDate", now)
        record.set("displayName", allergy)

    # Lab Results as Correlation / Quantity samples
    for lab in extraction.laboratory_results:
        record = ET.SubElement(root, "ClinicalRecord")
        record.set("type", "HKClinicalTypeIdentifierLabResultRecord")
        record.set("identifier", str(uuid.uuid4()))
        record.set("sourceRevision", "aInamnesis")
        record.set("startDate", lab.date or now)
        record.set("endDate", lab.date or now)
        record.set("displayName", lab.test)
        if lab.value:
            record.set("value", lab.value)
        if lab.unit:
            record.set("unit", lab.unit)
        if lab.reference_range:
            record.set("referenceRange", lab.reference_range)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(str(output_path), encoding="utf-8", xml_declaration=True)

    logger.info("Apple Health XML written: %s", output_path)
    return str(output_path)
