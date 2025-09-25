"""High-level orchestration for the FrequenCipher pipeline."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .anomaly import score_anomalies
from .backmask import detect_backmasking
from .ingestion import load_audio
from .models import AudioSignal
from .phase import detect_phase_anomalies
from .spectral import compute_spectral_features
from .steganography import detect_steganography
from .subliminal import detect_subliminal
from .temporal import check_temporal_manipulation
from .watermark import detect_watermark


AnalysisResult = Dict[str, Dict[str, Any]]


def _flatten_summaries(summaries: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    flattened: Dict[str, float] = {}
    for feature, stats in summaries.items():
        for stat_name, value in stats.items():
            flattened[f"{feature}_{stat_name}"] = float(value)
    return flattened


def run_full_analysis(
    path: str,
    *,
    target_sr: Optional[int] = 44100,
    mono: bool = True,
    include_raw_spectra: bool = False,
) -> Tuple[AudioSignal, AnalysisResult]:
    """Run the full analysis pipeline on the provided audio file."""

    audio = load_audio(path, target_sr=target_sr, mono=mono)
    samples = audio.samples
    sr = audio.sample_rate

    mono_samples = samples if samples.ndim == 1 else np.mean(samples, axis=0)
    spectral = compute_spectral_features(mono_samples, sr)
    spectral_result: Dict[str, Any] = {
        "summaries": spectral["summaries"],
    }
    if include_raw_spectra:
        spectral_result["matrices"] = {k: v.tolist() for k, v in spectral["matrices"].items()}

    phase = detect_phase_anomalies(mono_samples, sr)
    backmask = detect_backmasking(samples, sr)
    subliminal = detect_subliminal(samples, sr)
    stego = detect_steganography(samples, sr)
    temporal = check_temporal_manipulation(samples, sr)
    watermark = detect_watermark(samples, sr)

    flattened = _flatten_summaries(spectral["summaries"])
    anomaly_features = np.array(list(flattened.values())).reshape(1, -1)
    anomaly = score_anomalies(anomaly_features)

    results: AnalysisResult = {
        "metadata": {
            "sample_rate": audio.sample_rate,
            "duration_seconds": audio.duration,
            "channels": audio.channels,
        },
        "spectral": spectral_result,
        "phase": phase,
        "backmask": backmask,
        "subliminal": subliminal,
        "steganography": stego,
        "temporal": temporal,
        "watermark": watermark,
        "anomaly": anomaly,
    }

    return audio, results
