"""Phase-based hidden message detection."""
from __future__ import annotations

from typing import Dict

import librosa
import numpy as np
from scipy.stats import entropy

from .statistics import summarise_array


def detect_phase_anomalies(samples: np.ndarray, sample_rate: int) -> Dict[str, float]:
    """Analyse phase coherence to surface potential hidden encodings."""

    effective_fft = min(2048, samples.size if samples.size else 2048)
    if effective_fft < 2:
        effective_fft = 2
    stft = librosa.stft(samples, n_fft=effective_fft)
    phase = np.unwrap(np.angle(stft))
    phase_diff = np.diff(phase, axis=1)

    variance = float(np.var(phase_diff))
    mean_deviation = float(np.mean(np.abs(phase_diff)))
    coherence = float(np.abs(np.mean(np.exp(1j * phase_diff))))

    hist, _ = np.histogram(phase_diff.flatten(), bins=64, density=True)
    phase_entropy = float(entropy(hist + 1e-8))

    summary = summarise_array(phase_diff.flatten()).to_dict()

    return {
        "phase_variance": variance,
        "phase_mean_deviation": mean_deviation,
        "phase_coherence": coherence,
        "phase_entropy": phase_entropy,
        "phase_diff_mean": summary["mean"],
        "phase_diff_std": summary["std"],
        "phase_diff_percentile_75": summary["percentile_75"],
    }
