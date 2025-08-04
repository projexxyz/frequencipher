"""
Automated forensic report generation.
"""
from typing import Dict
import json
from fpdf import FPDF


def generate_report(results: Dict[str, Dict[str, float]], output_path: str) -> None:
    """Generate a simple PDF report from results.

    Parameters
    ----------
    results : dict
        Dictionary of analysis results keyed by module.
    output_path : str
        Where to save the PDF report.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, txt="FrequenCipher Forensic Report", ln=True, align='C')
    for module, metrics in results.items():
        pdf.ln(5)
        pdf.set_font("Arial", size=10, style='B')
        pdf.cell(0, 10, txt=module.capitalize(), ln=True)
        pdf.set_font("Arial", size=10)
        for key, value in metrics.items():
            pdf.cell(0, 8, txt=f"{key}: {value:.5f}", ln=True)
    pdf.output(output_path)
