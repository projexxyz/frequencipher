"""Steganography pattern recognition."""
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.stats import chisquare

from .statistics import summarise_array


def detect_steganography(samples: np.ndarray, sample_rate: int) -> Dict[str, float]:
    """Search for low-complexity steganographic alterations in the waveform."""

    if samples.ndim > 1:
        samples = np.mean(samples, axis=0)

    scaled = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
    lsb = scaled & 1
    transitions = np.mean(lsb[:-1] != lsb[1:]) if lsb.size > 1 else 0.0

    counts = np.bincount(lsb, minlength=2) if lsb.size else np.array([0, 0])
    if lsb.size:
        expected = np.array([lsb.size / 2, lsb.size / 2])
        chi2, _ = chisquare(counts, expected)
        chi2 = float(chi2)
    else:
        chi2 = 0.0

    second_lsb = (scaled >> 1) & 1
    correlation = float(np.corrcoef(lsb, second_lsb)[0, 1]) if lsb.size > 1 else 0.0
    if not np.isfinite(correlation):
        correlation = 0.0

    window_size = 2048
    num_windows = lsb.size // window_size
    if num_windows:
        windows = lsb[: num_windows * window_size].reshape(num_windows, window_size)
        window_summary = summarise_array(np.mean(windows, axis=1)).to_dict()
        window_var = window_summary["std"]
    else:
        window_var = 0.0

    return {
        "lsb_chi_square": chi2,
        "lsb_transition_rate": float(transitions),
        "lsb_second_bit_correlation": correlation,
        "lsb_window_std": float(window_var),
    }
