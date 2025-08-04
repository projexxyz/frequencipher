"""
Interactive visualization and dashboards (placeholder).
"""
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def show_spectrogram(y: np.ndarray, sr: int) -> None:
    """Display a mel spectrogram of the audio signal."""
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(S), sr=sr, x_axis='time', y_axis='mel')
    plt.title('Mel Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()
