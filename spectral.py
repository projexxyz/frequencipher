"""
Spectral and frequency domain decomposition using STFT, mel-spectrogram, MFCC, chroma, and wavelets.
"""
import numpy as np
import librosa
from typing import Dict


def compute_spectrogram(y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
    """Compute various spectral representations of audio.

    Returns a dictionary with keys: 'stft', 'mel', 'mfcc', 'chroma'.
    """
    stft = np.abs(librosa.stft(y))
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    return {"stft": stft, "mel": mel, "mfcc": mfcc, "chroma": chroma}
