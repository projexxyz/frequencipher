"""
Phase-based hidden message detection.
"""
import numpy as np
import librosa
from typing import Dict


def detect_phase_anomalies(y: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Detect irregularities in phase to flag potential hidden messages.

    This function computes the Short‑Time Fourier Transform (STFT) of the signal
    and then examines the phase component for inconsistencies. In normal audio,
    phase differences between neighbouring frames should vary smoothly. Sudden
    jumps or high variance can indicate phase‑encoded steganography.

    Parameters
    ----------
    y : np.ndarray
        Input time series (mono)
    sr : int
        Sampling rate of ``y``

    Returns
    -------
    dict
        Contains a single key ``phase_anomaly_score`` with a float value
        representing the variance of phase differences.
    """
    # Compute complex STFT
    stft = librosa.stft(y)
    # Extract the phase of each complex coefficient
    phase = np.angle(stft)
    # Compute first derivative along time axis (frame‑to‑frame phase change)
    phase_diff = np.diff(phase, axis=1)
    # Calculate variance of the phase difference as anomaly metric
    score = float(np.var(phase_diff))
    return {"phase_anomaly_score": score}
