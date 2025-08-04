# FrequenCipher

FrequenCipher is a high-level audio forensics toolkit designed to uncover covert and harmful content hidden within audio recordings. It provides a modular Python package with support for multiple file formats, spectral analysis, phase anomaly detection, backmasking analysis, subliminal frequency extraction, anomaly scoring, steganography and watermark detection, temporal manipulation checks, report generation, and visualization.

**Note:** This is a **skeleton implementation** intended for demonstration purposes. The core logic of each module must be developed and validated by audio forensics experts before use in real investigations.

## Features

- **Multi-format Ingestion:** Reads WAV, MP3, FLAC, OGG via `soundfile`.
- **Spectral Analysis:** Computes STFT, mel-spectrogram, MFCCs, and chroma features.
- **Phase Analysis:** Detects irregularities in phase to flag hidden messages.
- **Backmasking Detection:** Compares forward and reversed audio for hidden speech patterns.
- **Subliminal Detection:** Calculates energy in infrasonic and ultrasonic bands.
- **Anomaly Scoring:** Placeholder statistical method for identifying unusual audio sections.
- **Steganography & Watermark Detection:** Stub functions for detecting LSB manipulation and watermarking.
- **Temporal Checks:** Evaluates simple pitch stability metrics to detect speed/pitch shifts.
- **Report Generation:** Generates a PDF summary of analysis results.
- **CLI:** Run analyses from the command line.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python -m frequencipher.cli path/to/audio.wav --report output.pdf
```

## Disclaimer

This project is provided as a starting point. For real forensic applications, you must implement robust algorithms and obtain expert validation. Use responsibly and consult legal counsel regarding admissibility and privacy considerations.
