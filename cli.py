"""
Command-line interface for running FrequenCipher analyses.
"""
import argparse
from .ingestion import load_audio
from .spectral import compute_spectrogram
from .phase import detect_phase_anomalies
from .backmask import detect_backmasking
from .subliminal import detect_subliminal
from .steganography import detect_steganography
from .temporal import check_temporal_manipulation
from .report import generate_report


def main():
    parser = argparse.ArgumentParser(description="Run FrequenCipher forensic analysis on an audio file.")
    parser.add_argument("input", help="Path to the input audio file")
    parser.add_argument("--report", default="report.pdf", help="Output PDF report filename")
    args = parser.parse_args()
    y, sr = load_audio(args.input)
    results = {}
    results['spectral'] = {k: v.mean() for k, v in compute_spectrogram(y, sr).items()}  # summarize
    results['phase'] = detect_phase_anomalies(y, sr)
    results['backmask'] = detect_backmasking(y, sr)
    results['subliminal'] = detect_subliminal(y, sr)
    results['steganography'] = detect_steganography(y, sr)
    results['temporal'] = check_temporal_manipulation(y, sr)
    generate_report(results, args.report)
    print(f"Report saved to {args.report}")

if __name__ == '__main__':
    main()
