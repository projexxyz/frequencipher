"""Spectral and frequency-domain analysis utilities."""
from __future__ import annotations

from typing import Dict

import librosa
import numpy as np

from .statistics import summarise_array


def _compute_summary(matrix: np.ndarray) -> Dict[str, float]:
    return summarise_array(matrix.flatten()).to_dict()


def compute_spectral_features(
    samples: np.ndarray,
    sample_rate: int,
    *,
    n_fft: int = 2048,
    hop_length: int = 512,
    mel_bands: int = 128,
    include_wavelet: bool = False,
) -> Dict[str, Dict[str, Dict[str, float] | np.ndarray]]:
    """Compute a suite of spectral representations and their summaries."""

    if samples.size == 0:
        raise ValueError("Input samples must be non-empty")

    effective_fft = min(n_fft, samples.size)
    if effective_fft < 2:
        effective_fft = 2
    effective_hop = min(hop_length, max(1, effective_fft // 4))

    stft = np.abs(librosa.stft(samples, n_fft=effective_fft, hop_length=effective_hop))
    mel = librosa.feature.melspectrogram(
        y=samples,
        sr=sample_rate,
        n_fft=effective_fft,
        hop_length=effective_hop,
        n_mels=mel_bands,
    )
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel, ref=np.max), sr=sample_rate)
    chroma = librosa.feature.chroma_stft(
        y=samples,
        sr=sample_rate,
        n_fft=effective_fft,
        hop_length=effective_hop,
    )
    centroid = librosa.feature.spectral_centroid(
        y=samples,
        sr=sample_rate,
        n_fft=effective_fft,
        hop_length=effective_hop,
    )
    bandwidth = librosa.feature.spectral_bandwidth(
        y=samples,
        sr=sample_rate,
        n_fft=effective_fft,
        hop_length=effective_hop,
    )
    contrast = librosa.feature.spectral_contrast(
        y=samples,
        sr=sample_rate,
        n_fft=effective_fft,
        hop_length=effective_hop,
    )
    rolloff = librosa.feature.spectral_rolloff(
        y=samples,
        sr=sample_rate,
        n_fft=effective_fft,
        hop_length=effective_hop,
    )

    matrices: Dict[str, np.ndarray] = {
        "stft": stft,
        "mel": mel,
        "mfcc": mfcc,
        "chroma": chroma,
        "centroid": centroid,
        "bandwidth": bandwidth,
        "contrast": contrast,
        "rolloff": rolloff,
    }

    if include_wavelet:
        try:
            import pywt  # type: ignore

            coeffs = pywt.wavedec(samples, "db4", level=4)
            matrices["wavelet_coefficients"] = np.concatenate([c.flatten() for c in coeffs])
        except Exception:  # pragma: no cover - optional dependency
            pass

    summaries = {name: _compute_summary(matrix) for name, matrix in matrices.items()}

    return {
        "matrices": matrices,
        "summaries": summaries,
    }
