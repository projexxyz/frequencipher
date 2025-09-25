"""Detect subliminal frequency and psychoacoustic content."""
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.signal import hilbert

from .statistics import summarise_array


def detect_subliminal(samples: np.ndarray, sample_rate: int) -> Dict[str, float]:
    """Identify subliminal content via spectral and modulation cues."""

    if samples.ndim > 1:
        samples = np.mean(samples, axis=0)

    spectrum = np.fft.rfft(samples)
    freqs = np.fft.rfftfreq(samples.size, 1 / sample_rate)

    infra_mask = freqs < 20
    ultra_mask = freqs > 20000
    audible_mask = (~infra_mask) & (~ultra_mask)

    infra_energy = float(np.mean(np.abs(spectrum[infra_mask])) if np.any(infra_mask) else 0.0)
    ultra_energy = float(np.mean(np.abs(spectrum[ultra_mask])) if np.any(ultra_mask) else 0.0)
    audible_energy = float(np.mean(np.abs(spectrum[audible_mask])) if np.any(audible_mask) else 0.0)
    energy_ratio = float((infra_energy + ultra_energy) / (audible_energy + 1e-8))

    analytic = hilbert(samples)
    amplitude_envelope = np.abs(analytic)
    amplitude_summary = summarise_array(amplitude_envelope).to_dict()
    instantaneous_phase = np.unwrap(np.angle(analytic))
    instantaneous_freq = np.diff(instantaneous_phase) / (2.0 * np.pi) * sample_rate
    freq_summary = summarise_array(instantaneous_freq).to_dict() if instantaneous_freq.size else {
        "mean": 0.0,
        "std": 0.0,
        "median": 0.0,
        "min": 0.0,
        "max": 0.0,
        "percentile_25": 0.0,
        "percentile_75": 0.0,
    }

    return {
        "infrasound_energy": infra_energy,
        "ultrasound_energy": ultra_energy,
        "audible_energy": audible_energy,
        "subliminal_energy_ratio": energy_ratio,
        "amplitude_modulation_std": amplitude_summary["std"],
        "amplitude_modulation_percentile_75": amplitude_summary["percentile_75"],
        "frequency_modulation_std": freq_summary["std"],
        "frequency_modulation_percentile_75": freq_summary["percentile_75"],
    }
