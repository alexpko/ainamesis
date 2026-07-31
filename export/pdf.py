"""
pdf.py
Export a MedicalExtraction as a formatted PDF report using ReportLab.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_REPORTLAB_AVAILABLE = False
try:
    from reportlab.lib import colors  # type: ignore
    from reportlab.lib.pagesizes import A4  # type: ignore
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle  # type: ignore
    from reportlab.lib.units import cm  # type: ignore
    from reportlab.platypus import (  # type: ignore
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    )
    _REPORTLAB_AVAILABLE = True
except ImportError:
    logger.warning("reportlab not installed. PDF export is disabled.")


def export_to_pdf(extraction, output_path: str | Path | None = None) -> str:
    """
    Render a MedicalExtraction as a PDF.

    Parameters
    ----------
    extraction:
        ``MedicalExtraction`` instance from the LLM module.
    output_path:
        Destination path.  Defaults to ``exports/report_<timestamp>.pdf``.

    Returns
    -------
    str
        Absolute path to the written PDF file.

    Raises
    ------
    ImportError
        When reportlab is not installed.
    """
    if not _REPORTLAB_AVAILABLE:
        raise ImportError(
            "reportlab is required for PDF export. "
            "Install it with: pip install reportlab"
        )

    if output_path is None:
        exports_dir = Path.cwd() / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = exports_dir / f"report_{timestamp}.pdf"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    heading_style = ParagraphStyle(
        "Heading1Custom",
        parent=styles["Heading1"],
        fontSize=14,
        textColor=colors.HexColor("#1a3c5e"),
    )
    body_style = styles["Normal"]
    body_style.fontSize = 11

    story = []

    # Title
    story.append(Paragraph("aInamnesis — Medical Report", heading_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", body_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    story.append(Spacer(1, 0.4 * cm))

    # Diagnosis
    _append_section(story, "Diagnosis", extraction.diagnosis, heading_style, body_style)

    # Allergies
    _append_section(story, "Allergies", extraction.allergies, heading_style, body_style)

    # Medications table
    if extraction.medications:
        story.append(Paragraph("Medications", heading_style))
        table_data = [["Medication", "Dose", "Frequency", "Route"]]
        for med in extraction.medications:
            table_data.append([
                med.name,
                med.dose or "—",
                med.frequency or "—",
                med.route or "—",
            ])
        tbl = Table(table_data, colWidths=[6 * cm, 4 * cm, 4 * cm, 4 * cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3c5e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f4f8")]),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 0.4 * cm))

    # Laboratory results table
    if extraction.laboratory_results:
        story.append(Paragraph("Laboratory Results", heading_style))
        table_data = [["Test", "Value", "Unit", "Reference", "Date"]]
        for lab in extraction.laboratory_results:
            table_data.append([
                lab.test,
                lab.value or "—",
                lab.unit or "—",
                lab.reference_range or "—",
                lab.date or "—",
            ])
        tbl = Table(table_data, colWidths=[5 * cm, 3 * cm, 2.5 * cm, 4 * cm, 3.5 * cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3c5e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f4f8")]),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 0.4 * cm))

    # Timeline
    if extraction.timeline:
        story.append(Paragraph("Clinical Timeline", heading_style))
        for event in extraction.timeline:
            date_str = event.date or "Unknown date"
            story.append(Paragraph(f"<b>{date_str}</b>: {event.event}", body_style))
        story.append(Spacer(1, 0.4 * cm))

    doc = SimpleDocTemplate(str(output_path), pagesize=A4, rightMargin=2 * cm, leftMargin=2 * cm)
    doc.build(story)
    logger.info("PDF report written: %s", output_path)
    return str(output_path)


def _append_section(story, title: str, items: list, heading_style, body_style) -> None:
    story.append(Paragraph(title, heading_style))
    if items:
        for item in items:
            story.append(Paragraph(f"• {item}", body_style))
    else:
        story.append(Paragraph("None recorded.", body_style))
    story.append(Spacer(1, 0.3 * cm))
