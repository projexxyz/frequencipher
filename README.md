# FrequenCipher

FrequenCipher is an enterprise-grade audio forensics toolkit designed to uncover covert and harmful content hidden within audio recordings. It provides a modular Python package with support for multi-format ingestion, high fidelity spectral analysis, anomaly detection, backmasking inspection, subliminal frequency extraction, steganography and watermark heuristics, temporal manipulation checks, report generation, and a hardened CLI.

## Key capabilities

- **Robust ingestion pipeline:** Validates input files, supports streaming reads, automatic resampling, mono/stereo handling, and DC offset removal.
- **Rich spectral profiling:** Computes STFT, mel bands, MFCCs, chroma, spectral shape descriptors, and optional wavelet coefficients with statistical summaries for downstream ML models.
- **Phase anomaly scanning:** Quantifies phase coherence, entropy, and deviation metrics to reveal phase-encoded messages.
- **Backmasking heuristics:** Evaluates cross-correlation, frame-wise similarity, and energy symmetry between forward and reversed audio.
- **Subliminal & psychoacoustic metrics:** Measures infrasonic/ultrasonic energy, amplitude/frequency modulation strength, and highlights suspicious energy ratios.
- **Steganography fingerprinting:** Runs chi-square tests, transition analysis, and bit-plane correlations for simple LSB manipulations.
- **Temporal integrity checks:** Tracks tempo stability and zero-crossing behaviour to flag speed or pitch tampering.
- **Watermark detection:** Uses spectral flatness and tonal centroid dispersion to score hidden watermark likelihood.
- **Enterprise reporting:** Generates structured PDF and JSON artefacts, with logging and automation-friendly CLI options.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python -m frequencipher.cli path/to/audio.wav --report analysis.pdf --json analysis.json
```

### CLI options

| Option | Description |
| --- | --- |
| `--report` | Path to write a PDF report (optional). |
| `--json` | Path to export raw JSON results (optional). |
| `--target-sr` | Resample audio to the specified rate before analysis (default `44100`). |
| `--stereo` | Preserve stereo channels (default downmix to mono). |
| `--include-raw-spectra` | Include raw spectral matrices in JSON output. |
| `--log-level` | Configure logging verbosity (default `INFO`). |

## Programmatic usage

```python
from frequencipher import run_full_analysis

audio, results = run_full_analysis("suspect.wav", target_sr=48000)
print(audio.to_dict())
print(results["spectral"]["summaries"]["mfcc"])  # access summary stats
```

## Disclaimer

This project ships with heuristic algorithms designed for triage and investigative support. Always validate results against expert judgement and jurisdiction-specific legal guidance before acting on findings.
