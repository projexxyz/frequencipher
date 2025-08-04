"""
Detect subliminal frequency and psychoacoustic content.
"""
import numpy as np
from typing import Dict


def detect_subliminal(y: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Identify subliminal content by examining energy in frequency bands outside the
    typical hearing range and by inspecting amplitude and frequency modulation.

    Parameters
    ----------
    y : np.ndarray
        Input signal
    sr : int
        Sampling rate

    Returns
    -------
    dict
        Metrics including infrasound and ultrasound energy, amplitude modulation
        strength, and frequency modulation strength.
    """
    # Frequency domain analysis for infrasound (<20 Hz) and ultrasound (>20 kHz)
    spectrum = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(len(y), 1 / sr)
    infra_energy = float(np.mean(np.abs(spectrum[freqs < 20])))
    ultra_energy = float(np.mean(np.abs(spectrum[freqs > 20000])))
    # Amplitude modulation detection: compute envelope via Hilbert transform
    from scipy.signal import hilbert
    analytic_signal = hilbert(y)
    amplitude_envelope = np.abs(analytic_signal)
    # Measure variance of envelope to assess modulation depth
    am_strength = float(np.var(amplitude_envelope))
    # Frequency modulation detection: compare instantaneous frequency variance
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_freq = (np.diff(instantaneous_phase) / (2.0 * np.pi) * sr)
    fm_strength = float(np.var(instantaneous_freq)) if len(instantaneous_freq) > 1 else 0.0
    return {
        "infrasound_energy": infra_energy,
        "ultrasound_energy": ultra_energy,
        "amplitude_modulation_strength": am_strength,
        "frequency_modulation_strength": fm_strength,
    }
