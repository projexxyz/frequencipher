"""Audio watermark detection module."""
from __future__ import annotations

from typing import Dict

import librosa
import numpy as np
import warnings

from .statistics import summarise_array


def detect_watermark(samples: np.ndarray, sample_rate: int) -> Dict[str, float]:
    """Heuristic watermark detection based on spectral flatness and tonality."""

    if samples.ndim > 1:
        samples = np.mean(samples, axis=0)

    effective_fft = min(2048, samples.size if samples.size else 2048)
    if effective_fft < 2:
        effective_fft = 2
    stft = np.abs(librosa.stft(samples, n_fft=effective_fft))
    spectral_flatness = librosa.feature.spectral_flatness(S=stft)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*n_fft=.*is too large.*", category=UserWarning)
        tonal_centroid = librosa.feature.tonnetz(y=samples, sr=sample_rate)

    flatness_summary = summarise_array(spectral_flatness.flatten()).to_dict()
    tonal_summary = summarise_array(tonal_centroid.flatten()).to_dict()

    watermark_score = float(
        1.0 - min(flatness_summary["mean"], 1.0)
        + max(0.0, tonal_summary["std"] - 0.1)
    )

    return {
        "spectral_flatness_mean": flatness_summary["mean"],
        "spectral_flatness_std": flatness_summary["std"],
        "tonal_centroid_std": tonal_summary["std"],
        "watermark_score": watermark_score,
    }
