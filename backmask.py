"""
Real-time backmasking detection via reversed audio analysis.
"""
import numpy as np
from typing import Dict


def detect_backmasking(y: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Detect potential backmasked speech by comparing forward and reversed audio.

    The algorithm reverses the signal and computes a similarity metric between
    segments of the original and reversed audio. A high similarity score may
    indicate intentional embedding of intelligible content when played backwards.

    Parameters
    ----------
    y : np.ndarray
        Input time series
    sr : int
        Sampling rate

    Returns
    -------
    dict
        A dictionary with a single key ``backmasking_score``. Scores closer to 1
        indicate stronger correlation between forward and reversed audio, which
        can be suspicious.
    """
    # Reverse the audio
    reversed_y = y[::-1]
    # Normalize both signals
    if np.max(np.abs(y)) > 0:
        y_norm = y / np.max(np.abs(y))
    else:
        y_norm = y
    if np.max(np.abs(reversed_y)) > 0:
        rev_norm = reversed_y / np.max(np.abs(reversed_y))
    else:
        rev_norm = reversed_y
    # Compute correlation coefficient
    min_len = min(len(y_norm), len(rev_norm))
    corr = np.correlate(y_norm[:min_len], rev_norm[:min_len], mode='valid')[0]
    score = float(corr / min_len)
    return {"backmasking_score": score}
