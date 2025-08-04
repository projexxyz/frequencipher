"""
Audio ingestion and preprocessing module.
Supports reading various file formats and normalizing the audio signal.
"""
from typing import Tuple
import soundfile as sf
import numpy as np

SUPPORTED_FORMATS = {"wav", "mp3", "flac", "ogg"}

def load_audio(path: str) -> Tuple[np.ndarray, int]:
    """Load an audio file and return a normalized signal and sample rate.

    Parameters
    ----------
    path : str
        Path to the audio file.

    Returns
    -------
    Tuple[np.ndarray, int]
        Tuple of audio time series and sample rate.
    """
    data, sr = sf.read(path)
    # Normalize to [-1, 1]
    if data.dtype != np.float32 and data.dtype != np.float64:
        data = data / np.iinfo(data.dtype).max
    return data.astype(float), sr
