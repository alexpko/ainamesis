"""
test_export.py
Unit tests for the export module (FHIR, Apple Health, PDF).
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from llm.ollama import MedicalExtraction, MedicationEntry, LabResult, TimelineEvent


@pytest.fixture
def full_extraction():
    return MedicalExtraction(
        diagnosis=["Type 2 Diabetes Mellitus", "Hypertension"],
        medications=[
            MedicationEntry(name="Metformin", dose="500mg", frequency="twice daily", route="oral"),
            MedicationEntry(name="Amlodipine", dose="5mg", frequency="once daily", route=None),
        ],
        allergies=["Penicillin", "Sulfa"],
        laboratory_results=[
            LabResult(test="HbA1c", value="7.2", unit="%", reference_range="4.0-6.0", date="2024-03-01"),
            LabResult(test="Creatinine", value="1.1", unit="mg/dL", reference_range=None, date=None),
        ],
        timeline=[
            TimelineEvent(date="2024-01-15", event="First presentation"),
            TimelineEvent(date=None, event="Patient reports fatigue"),
        ],
    )


@pytest.fixture
def empty_extraction():
    return MedicalExtraction()


# ---------------------------------------------------------------------------
# FHIR export
# ---------------------------------------------------------------------------

class TestFHIRExport:
    def test_export_creates_file(self, full_extraction, tmp_path):
        from export.fhir import export_to_fhir
        path = export_to_fhir(full_extraction, output_path=tmp_path / "out.json")
        assert Path(path).exists()

    def test_bundle_structure(self, full_extraction, tmp_path):
        from export.fhir import export_to_fhir
        path = export_to_fhir(full_extraction, output_path=tmp_path / "out.json")
        with open(path) as f:
            data = json.load(f)
        assert data["resourceType"] == "Bundle"
        assert data["type"] == "collection"
        assert "entry" in data
        assert isinstance(data["entry"], list)

    def test_entry_count(self, full_extraction, tmp_path):
        """Each diagnosis, medication, allergy, and lab result → 1 entry each."""
        from export.fhir import export_to_fhir
        path = export_to_fhir(full_extraction, output_path=tmp_path / "out.json")
        with open(path) as f:
            data = json.load(f)
        # 2 diagnosis + 2 medications + 2 allergies + 2 lab results = 8
        assert len(data["entry"]) == 8

    def test_resource_types_present(self, full_extraction, tmp_path):
        from export.fhir import export_to_fhir
        path = export_to_fhir(full_extraction, output_path=tmp_path / "out.json")
        with open(path) as f:
            data = json.load(f)
        resource_types = {e["resource"]["resourceType"] for e in data["entry"]}
        assert "Condition" in resource_types
        assert "MedicationStatement" in resource_types
        assert "AllergyIntolerance" in resource_types
        assert "Observation" in resource_types

    def test_empty_extraction_produces_empty_bundle(self, empty_extraction, tmp_path):
        from export.fhir import export_to_fhir
        path = export_to_fhir(empty_extraction, output_path=tmp_path / "out.json")
        with open(path) as f:
            data = json.load(f)
        assert data["entry"] == []

    def test_default_output_path(self, full_extraction, tmp_path, monkeypatch):
        from export import fhir as fhir_module
        monkeypatch.chdir(tmp_path)
        from export.fhir import export_to_fhir
        path = export_to_fhir(full_extraction)
        assert Path(path).exists()


# ---------------------------------------------------------------------------
# Apple Health export
# ---------------------------------------------------------------------------

class TestAppleHealthExport:
    def test_export_creates_file(self, full_extraction, tmp_path):
        from export.apple_health import export_to_apple_health
        path = export_to_apple_health(full_extraction, output_path=tmp_path / "out.xml")
        assert Path(path).exists()

    def test_valid_xml(self, full_extraction, tmp_path):
        from export.apple_health import export_to_apple_health
        path = export_to_apple_health(full_extraction, output_path=tmp_path / "out.xml")
        tree = ET.parse(path)
        root = tree.getroot()
        assert root.tag == "HealthData"

    def test_clinical_records_present(self, full_extraction, tmp_path):
        from export.apple_health import export_to_apple_health
        path = export_to_apple_health(full_extraction, output_path=tmp_path / "out.xml")
        tree = ET.parse(path)
        root = tree.getroot()
        records = root.findall("ClinicalRecord")
        # 2 diagnosis + 2 medications + 2 allergies + 2 labs = 8
        assert len(records) == 8

    def test_medication_dose_attribute(self, full_extraction, tmp_path):
        from export.apple_health import export_to_apple_health
        path = export_to_apple_health(full_extraction, output_path=tmp_path / "out.xml")
        tree = ET.parse(path)
        root = tree.getroot()
        med_records = [
            r for r in root.findall("ClinicalRecord")
            if r.get("type") == "HKClinicalTypeIdentifierMedicationRecord"
        ]
        assert any(r.get("dose") == "500mg" for r in med_records)

    def test_empty_extraction(self, empty_extraction, tmp_path):
        from export.apple_health import export_to_apple_health
        path = export_to_apple_health(empty_extraction, output_path=tmp_path / "out.xml")
        tree = ET.parse(path)
        root = tree.getroot()
        assert root.findall("ClinicalRecord") == []
