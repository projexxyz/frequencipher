"""Command-line interface for running FrequenCipher analyses."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from .analysis import run_full_analysis
from .exceptions import FrequenCipherError
from .report import generate_report


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FrequenCipher forensic analysis on an audio file.")
    parser.add_argument("input", help="Path to the input audio file")
    parser.add_argument("--report", help="Output PDF report filename", default=None)
    parser.add_argument("--json", help="Optional path to dump raw JSON results", default=None)
    parser.add_argument("--target-sr", type=int, default=44100, help="Target sample rate for analysis")
    parser.add_argument("--stereo", action="store_true", help="Preserve stereo channels instead of down-mixing")
    parser.add_argument("--include-raw-spectra", action="store_true", help="Include raw spectral matrices in JSON output")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    return parser.parse_args()


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    try:
        audio, results = run_full_analysis(
            args.input,
            target_sr=args.target_sr,
            mono=not args.stereo,
            include_raw_spectra=args.include_raw_spectra,
        )
    except FrequenCipherError as exc:
        logging.error("Analysis failed: %s", exc)
        raise SystemExit(1) from exc

    logging.info(
        "Analysed '%s' (%.2f s @ %d Hz)",
        audio.path,
        audio.duration,
        audio.sample_rate,
    )

    if args.json:
        _dump_json(Path(args.json), results)
        logging.info("Wrote JSON results to %s", args.json)

    if args.report:
        generate_report(results, args.report)
        logging.info("Report saved to %s", args.report)


if __name__ == "__main__":
    main()
