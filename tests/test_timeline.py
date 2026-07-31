"""
test_timeline.py
Unit tests for the timeline module.
"""
from __future__ import annotations

from datetime import date

import pytest

from llm.ollama import MedicalExtraction, MedicationEntry, LabResult, TimelineEvent
from timeline.timeline import TimelineBuilder, TimelineEntry, MedicalTimeline


class TestDateParsing:
    def _parse(self, s):
        return TimelineBuilder._parse_date(s)

    def test_iso_date(self):
        assert self._parse("2024-03-15") == date(2024, 3, 15)

    def test_dd_mm_yyyy_slash(self):
        assert self._parse("15/03/2024") == date(2024, 3, 15)

    def test_dd_mm_yyyy_dot(self):
        assert self._parse("15.03.2024") == date(2024, 3, 15)

    def test_year_only(self):
        assert self._parse("2022") == date(2022, 1, 1)

    def test_none_input(self):
        assert self._parse(None) is None

    def test_empty_string(self):
        assert self._parse("") is None

    def test_unparseable(self):
        assert self._parse("yesterday") is None


class TestTimelineBuilder:
    def _make_extraction(self, timeline_events=None, lab_results=None):
        return MedicalExtraction(
            timeline=timeline_events or [],
            laboratory_results=lab_results or [],
        )

    def test_builds_from_llm_timeline(self):
        extraction = self._make_extraction(timeline_events=[
            TimelineEvent(date="2023-01-10", event="Hospital admission"),
            TimelineEvent(date="2023-02-01", event="Discharge"),
        ])
        timeline = TimelineBuilder().build(extraction)
        assert len(timeline.entries) == 2

    def test_sorted_entries_ascending(self):
        extraction = self._make_extraction(timeline_events=[
            TimelineEvent(date="2023-06-01", event="Second visit"),
            TimelineEvent(date="2022-01-15", event="First visit"),
        ])
        timeline = TimelineBuilder().build(extraction)
        sorted_entries = timeline.sorted_entries()
        assert sorted_entries[0].event == "First visit"
        assert sorted_entries[1].event == "Second visit"

    def test_unknown_dates_go_last(self):
        extraction = self._make_extraction(timeline_events=[
            TimelineEvent(date=None, event="Undated event"),
            TimelineEvent(date="2024-01-01", event="Known event"),
        ])
        timeline = TimelineBuilder().build(extraction)
        sorted_entries = timeline.sorted_entries()
        assert sorted_entries[-1].event == "Undated event"

    def test_lab_results_added_to_timeline(self):
        extraction = self._make_extraction(lab_results=[
            LabResult(test="HbA1c", value="7.2", unit="%", date="2024-03-01"),
        ])
        timeline = TimelineBuilder().build(extraction)
        assert any("HbA1c" in e.event for e in timeline.entries)
        assert timeline.entries[0].source == "lab"

    def test_deduplication(self):
        extraction = self._make_extraction(timeline_events=[
            TimelineEvent(date="2024-01-01", event="Fever"),
            TimelineEvent(date="2024-01-01", event="Fever"),  # duplicate
        ])
        timeline = TimelineBuilder().build(extraction)
        assert len(timeline.entries) == 1

    def test_to_dict_list(self):
        extraction = self._make_extraction(timeline_events=[
            TimelineEvent(date="2024-05-10", event="Checkup"),
        ])
        timeline = TimelineBuilder().build(extraction)
        result = timeline.to_dict_list()
        assert isinstance(result, list)
        assert result[0]["event"] == "Checkup"
        assert result[0]["date"] == "2024-05-10"

    def test_empty_extraction(self):
        extraction = self._make_extraction()
        timeline = TimelineBuilder().build(extraction)
        assert timeline.entries == []


class TestTimelineEntry:
    def test_display_date_with_parsed(self):
        entry = TimelineEntry(raw_date="2024-01-01", parsed_date=date(2024, 1, 1), event="Test")
        assert entry.display_date() == "2024-01-01"

    def test_display_date_with_raw_only(self):
        entry = TimelineEntry(raw_date="early 2023", parsed_date=None, event="Test")
        assert entry.display_date() == "early 2023"

    def test_display_date_no_date(self):
        entry = TimelineEntry(raw_date=None, parsed_date=None, event="Test")
        assert entry.display_date() == "Unknown date"
