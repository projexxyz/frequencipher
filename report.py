"""Automated forensic report generation."""
from __future__ import annotations

from typing import Any, Dict

from fpdf import FPDF


def _render_value(pdf: FPDF, key: str, value: Any, indent: int = 0) -> None:
    prefix = " " * indent
    if isinstance(value, dict):
        pdf.set_font("Arial", size=10, style='B')
        pdf.cell(0, 8, txt=f"{prefix}{key}", ln=True)
        pdf.set_font("Arial", size=10)
        for sub_key, sub_value in value.items():
            _render_value(pdf, str(sub_key), sub_value, indent + 2)
    elif isinstance(value, (list, tuple)):
        pdf.cell(0, 8, txt=f"{prefix}{key}:", ln=True)
        for item in value:
            _render_value(pdf, "-", item, indent + 2)
    else:
        pdf.cell(0, 8, txt=f"{prefix}{key}: {value}", ln=True)


def generate_report(results: Dict[str, Dict[str, Any]], output_path: str) -> None:
    """Generate a structured PDF report from analysis results."""

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=14, style='B')
    pdf.cell(0, 12, txt="FrequenCipher Forensic Report", ln=True, align='C')
    pdf.ln(4)

    for module, metrics in results.items():
        pdf.set_font("Arial", size=12, style='B')
        pdf.cell(0, 10, txt=module.replace("_", " ").title(), ln=True)
        pdf.set_font("Arial", size=10)
        for key, value in metrics.items():
            _render_value(pdf, str(key), value, indent=2)
        pdf.ln(2)

    pdf.output(output_path)
