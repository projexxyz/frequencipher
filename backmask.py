"""Real-time backmasking detection via reversed audio analysis."""
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.signal import correlate

from .statistics import summarise_array


def detect_backmasking(samples: np.ndarray, sample_rate: int) -> Dict[str, float]:
    """Detect potential backmasked speech by comparing forward and reversed audio."""

    if samples.ndim > 1:
        samples = np.mean(samples, axis=0)

    reversed_samples = samples[::-1]

    norm = np.linalg.norm(samples) * np.linalg.norm(reversed_samples)
    if norm == 0:
        peak_correlation = 0.0
    else:
        correlation = correlate(samples, reversed_samples, mode="valid")
        peak_correlation = float(np.max(np.abs(correlation)) / norm)

    frame_size = min(len(samples), sample_rate * 5)
    if frame_size == 0:
        energy_symmetry = 0.0
        summary = summarise_array(np.array([0.0])).to_dict()
    else:
        frames = len(samples) // frame_size or 1
        reshaped = samples[: frames * frame_size].reshape(frames, frame_size)
        reversed_reshaped = reversed_samples[: frames * frame_size].reshape(frames, frame_size)
        frame_correlations = []
        for i in range(frames):
            a = reshaped[i]
            b = reversed_reshaped[i]
            if np.std(a) == 0 or np.std(b) == 0:
                frame_correlations.append(0.0)
            else:
                frame_correlations.append(float(np.corrcoef(a, b)[0, 1]))
        summary = summarise_array(np.array(frame_correlations)).to_dict()
        energy_symmetry = float(
            np.mean(
                np.abs(np.sum(reshaped, axis=1) - np.sum(reversed_reshaped, axis=1))
            )
            / frame_size
        )

    return {
        "peak_correlation": peak_correlation,
        "frame_correlation_mean": summary["mean"],
        "frame_correlation_std": summary["std"],
        "energy_symmetry": energy_symmetry,
    }
