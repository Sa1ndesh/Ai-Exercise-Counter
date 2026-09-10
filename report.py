"""Session report generation: PDF and CSV.

`build_pdf_bytes` and `build_csv_bytes` return bytes so the same code can serve
a Streamlit download button or be written straight to disk by the CLI.
"""

import csv
import io
import os
import re
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from config import REPORT_DIR
from database import SessionRecord

# Column order used by both the CSV export and the history table.
CSV_COLUMNS = [
    "session_id",
    "student_id",
    "student_name",
    "exercise_type",
    "required_repetitions",
    "completed_repetitions",
    "valid_repetitions",
    "invalid_repetitions",
    "duration",
    "date_time",
    "status",
]

_ACCENT = colors.HexColor("#1F4E79")
_LIGHT = colors.HexColor("#EEF3F8")
_OK = colors.HexColor("#1B7F4B")
_BAD = colors.HexColor("#B3261E")


def _pretty_datetime(iso: str) -> str:
    """Render an ISO timestamp for humans, falling back to the raw string."""

    try:
        return datetime.fromisoformat(iso).strftime("%d %b %Y, %H:%M:%S")
    except (ValueError, TypeError):
        return iso


def report_rows(record: SessionRecord) -> list[tuple[str, str]]:
    """The specification's report fields, as label/value pairs."""

    return [
        ("Student ID", record.student_id),
        ("Student Name", record.student_name),
        ("Exercise Type", record.exercise_type),
        ("Required Repetitions", str(record.required_repetitions)),
        ("Completed Repetitions", str(record.completed_repetitions)),
        ("Valid Repetitions", str(record.valid_repetitions)),
        ("Invalid Repetitions", str(record.invalid_repetitions)),
        ("Form Accuracy", f"{record.accuracy}%"),
        ("Duration", f"{record.duration_text} ({record.duration:.1f}s)"),
        ("Completion Status", record.status),
        ("Date and Time", _pretty_datetime(record.date_time)),
    ]


def build_pdf_bytes(record: SessionRecord) -> bytes:
    """Render a one-page PDF report for a single session."""

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        title=f"Exercise Report - {record.student_name}",
        author="Student Exercise Monitoring System",
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        textColor=_ACCENT,
        fontSize=18,
        spaceAfter=2,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        textColor=colors.grey,
        fontSize=9,
    )

    story = [
        Paragraph("Student Exercise Session Report", title_style),
        Paragraph("AI-Based Student Exercise Monitoring System", subtitle_style),
        Spacer(1, 10 * mm),
    ]

    data = [["Field", "Value"]] + [list(row) for row in report_rows(record)]
    table = Table(data, colWidths=[65 * mm, 85 * mm], hAlign="CENTER")

    status_row = len(data) - 2  # "Completion Status" is second from last
    status_colour = _OK if record.status == "COMPLETED" else _BAD

    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), _ACCENT),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _LIGHT]),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B8C4D0")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                # Highlight the completion verdict.
                ("TEXTCOLOR", (1, status_row), (1, status_row), status_colour),
                ("FONTNAME", (1, status_row), (1, status_row), "Helvetica-Bold"),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 8 * mm))

    verdict = (
        f"Target of {record.required_repetitions} valid repetitions was reached."
        if record.status == "COMPLETED"
        else (
            f"Session ended {max(0, record.required_repetitions - record.valid_repetitions)} "
            f"valid repetitions short of the {record.required_repetitions} required."
        )
    )
    story.append(Paragraph(verdict, styles["Normal"]))
    story.append(Spacer(1, 4 * mm))
    story.append(
        Paragraph(
            "Repetitions are counted from MediaPipe Pose joint angles. A repetition is "
            "marked valid only when the required range of motion is achieved; shallow "
            "repetitions are recorded as invalid and do not count towards the target.",
            ParagraphStyle("Note", parent=styles["Normal"], fontSize=8, textColor=colors.grey),
        )
    )

    doc.build(story)
    return buffer.getvalue()


def build_csv_bytes(record: SessionRecord) -> bytes:
    """Single-session CSV: one header row and one data row."""

    return build_csv_bytes_many([record])


def build_csv_bytes_many(records: list[SessionRecord]) -> bytes:
    """Multi-session CSV, used by the history page's bulk export."""

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS + ["accuracy"], lineterminator="\n")
    writer.writeheader()

    for record in records:
        row = {column: getattr(record, column) for column in CSV_COLUMNS}
        row["duration"] = round(record.duration, 2)
        row["accuracy"] = record.accuracy
        writer.writerow(row)

    return buffer.getvalue().encode("utf-8")


def report_basename(record: SessionRecord) -> str:
    """Filesystem-safe stem for a session's report files."""

    safe_id = re.sub(r"[^A-Za-z0-9_-]+", "_", record.student_id) or "student"
    stamp = re.sub(r"[^0-9]", "", record.date_time)[:14]
    suffix = record.session_id if record.session_id is not None else stamp
    return f"session_{suffix}_{safe_id}_{record.exercise_type.lower()}"


def save_reports(record: SessionRecord, directory: str | None = None) -> dict[str, str]:
    """Write the PDF and CSV to disk. Returns the paths that were written."""

    target = directory or REPORT_DIR
    os.makedirs(target, exist_ok=True)
    stem = report_basename(record)

    pdf_path = os.path.join(target, f"{stem}.pdf")
    csv_path = os.path.join(target, f"{stem}.csv")

    with open(pdf_path, "wb") as handle:
        handle.write(build_pdf_bytes(record))
    with open(csv_path, "wb") as handle:
        handle.write(build_csv_bytes(record))

    return {"pdf": pdf_path, "csv": csv_path}


def format_console_report(record: SessionRecord) -> str:
    """Plain-text report for the command line interface."""

    width = 58
    lines = [
        "=" * width,
        "STUDENT EXERCISE SESSION REPORT".center(width),
        "=" * width,
    ]
    for label, value in report_rows(record):
        lines.append(f"  {label:<24}: {value}")
    lines.append("=" * width)
    return "\n".join(lines)
