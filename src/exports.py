"""Export analysis results as a downloadable PDF report."""

from __future__ import annotations

from io import BytesIO

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def build_pdf_report(job_title: str, job_count: int, skills: pd.DataFrame) -> bytes:
    """Create a compact PDF report for the selected role."""
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
    )
    styles = getSampleStyleSheet()
    content = [
        Paragraph("Tech Skills Recommender", styles["Title"]),
        Paragraph(f"Results for: {job_title}", styles["Heading2"]),
        Paragraph(
            f"Based on {job_count} matching historical job postings.",
            styles["BodyText"],
        ),
        Spacer(1, 14),
    ]

    rows = [["Keyword / candidate skill", "Job postings", "Share"]]
    rows.extend(
        [
            [row.skill, str(row.job_count), f"{row.share_percent:.0f}%"]
            for row in skills.itertuples(index=False)
        ]
    )
    table = Table(rows, colWidths=[3.8 * inch, 1.4 * inch, 1.2 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#183153")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F6FC")]),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D9E2F0")),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("PADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    content.extend(
        [
            table,
            Spacer(1, 14),
            Paragraph(
                "Source: xanderios/linkedin-job-postings on Hugging Face (dataset "
                "repository labeled MIT). These are historical postings, not live "
                "vacancies or current labour-market evidence.",
                styles["Italic"],
            ),
        ]
    )
    document.build(content)
    return buffer.getvalue()
