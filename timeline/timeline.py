"""
timeline.py
Extracts, sorts, and normalises a chronological medical event timeline
from a ``MedicalExtraction`` object produced by the LLM module.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import List, Optional

logger = logging.getLogger(__name__)

# Regex patterns for common date formats found in medical documents
_DATE_PATTERNS: list[tuple[str, str]] = [
    # YYYY-MM-DD
    (r"(\d{4})-(\d{2})-(\d{2})", "%Y-%m-%d"),
    # DD/MM/YYYY  or  MM/DD/YYYY (treated as DD/MM/YYYY by default)
    (r"(\d{2})/(\d{2})/(\d{4})", "%d/%m/%Y"),
    # DD.MM.YYYY
    (r"(\d{2})\.(\d{2})\.(\d{4})", "%d.%m.%Y"),
    # Month name, e.g. "January 2023" → treated as first of month
    (r"([A-Za-z]+)\s+(\d{4})", "%B %Y"),
    # Year only, e.g. "2022"
    (r"\b(\d{4})\b", "%Y"),
]


@dataclass
class TimelineEntry:
    """A single normalised timeline entry."""
    raw_date: Optional[str]
    parsed_date: Optional[date]
    event: str
    source: str = "llm"   # "llm" | "lab" | "manual"

    def display_date(self) -> str:
        if self.parsed_date:
            return self.parsed_date.isoformat()
        return self.raw_date or "Unknown date"


@dataclass
class MedicalTimeline:
    """Sorted, deduplicated collection of timeline entries."""
    entries: List[TimelineEntry] = field(default_factory=list)

    def sorted_entries(self) -> List[TimelineEntry]:
        """Return entries in ascending chronological order (unknowns last)."""
        known = [e for e in self.entries if e.parsed_date is not None]
        unknown = [e for e in self.entries if e.parsed_date is None]
        return sorted(known, key=lambda e: e.parsed_date) + unknown  # type: ignore[arg-type]

    def to_dict_list(self) -> list[dict]:
        return [
            {"date": e.display_date(), "event": e.event, "source": e.source}
            for e in self.sorted_entries()
        ]


class TimelineBuilder:
    """
    Builds a ``MedicalTimeline`` from a ``MedicalExtraction``.

    Usage::

        from llm.ollama import OllamaLLMClient
        from timeline.timeline import TimelineBuilder

        extraction = OllamaLLMClient().extract(ocr_text)
        timeline = TimelineBuilder().build(extraction)
        for entry in timeline.sorted_entries():
            print(entry.display_date(), "—", entry.event)
    """

    def build(self, extraction) -> MedicalTimeline:
        """
        Parameters
        ----------
        extraction:
            A ``MedicalExtraction`` instance.

        Returns
        -------
        MedicalTimeline
        """
        timeline = MedicalTimeline()

        # Events already tagged in the LLM timeline field
        for event in extraction.timeline:
            entry = TimelineEntry(
                raw_date=event.date,
                parsed_date=self._parse_date(event.date),
                event=event.event,
                source="llm",
            )
            timeline.entries.append(entry)

        # Enrich with dated laboratory results
        for lab in extraction.laboratory_results:
            if lab.date:
                desc = f"Lab: {lab.test}"
                if lab.value:
                    desc += f" = {lab.value}"
                    if lab.unit:
                        desc += f" {lab.unit}"
                entry = TimelineEntry(
                    raw_date=lab.date,
                    parsed_date=self._parse_date(lab.date),
                    event=desc,
                    source="lab",
                )
                timeline.entries.append(entry)

        self._deduplicate(timeline)
        logger.info("Timeline built: %d entries.", len(timeline.entries))
        return timeline

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_date(raw: Optional[str]) -> Optional[date]:
        """Attempt to parse a date string into a ``datetime.date`` object."""
        if not raw:
            return None
        raw = raw.strip()

        for pattern, fmt in _DATE_PATTERNS:
            m = re.search(pattern, raw)
            if m:
                candidate = m.group(0)
                try:
                    if fmt == "%Y":
                        return date(int(candidate), 1, 1)
                    if fmt == "%B %Y":
                        return datetime.strptime(candidate, fmt).date().replace(day=1)
                    return datetime.strptime(candidate, fmt).date()
                except ValueError:
                    continue
        logger.debug("Could not parse date: %r", raw)
        return None

    @staticmethod
    def _deduplicate(timeline: MedicalTimeline) -> None:
        """Remove exact-duplicate (date, event) pairs in place."""
        seen: set[tuple] = set()
        unique: List[TimelineEntry] = []
        for entry in timeline.entries:
            key = (entry.raw_date, entry.event.lower().strip())
            if key not in seen:
                seen.add(key)
                unique.append(entry)
        timeline.entries = unique
