"""Audio ingestion and preprocessing module."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import numpy.typing as npt

from .exceptions import AudioLoadingError, UnsupportedFormatError
from .models import AudioSignal

SUPPORTED_FORMATS = {"wav", "mp3", "flac", "ogg"}


def _validate_path(path: Path) -> None:
    if not path.exists():
        raise AudioLoadingError(f"Audio file '{path}' does not exist.")
    if not path.is_file():
        raise AudioLoadingError(f"Audio path '{path}' is not a regular file.")
    if path.suffix.lower().lstrip(".") not in SUPPORTED_FORMATS:
        raise UnsupportedFormatError(
            f"Unsupported audio format '{path.suffix}'. Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}."
        )


def _to_mono(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 1:
        return samples
    return np.mean(samples, axis=1)


def _remove_dc_offset(samples: np.ndarray) -> np.ndarray:
    return samples - float(np.mean(samples))


def _normalise(samples: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(samples))
    if peak == 0:
        return samples.astype(np.float32)
    return (samples / peak).astype(np.float32)


def _resample_if_needed(samples: np.ndarray, sr: int, target_sr: Optional[int]) -> Tuple[np.ndarray, int]:
    if target_sr is None or target_sr == sr:
        return samples, sr
    gcd = np.gcd(sr, target_sr)
    up = target_sr // gcd
    down = sr // gcd
    resampled = resample_poly(samples, up, down).astype(np.float32)
    return resampled, target_sr


def load_audio(
    path: str | Path,
    *,
    target_sr: Optional[int] = None,
    mono: bool = True,
    dtype: npt.DTypeLike = np.float32,
    chunk_size: Optional[int] = None,
) -> AudioSignal:
    """Load an audio file, returning an :class:`AudioSignal` instance.

    Parameters
    ----------
    path:
        Path to the audio file.
    target_sr:
        Optional sampling rate to resample the audio to using polyphase filtering.
    mono:
        If ``True`` (default) audio is downmixed to mono.
    dtype:
        Floating point dtype for the returned samples.
    chunk_size:
        Optional chunk size (in samples) to stream from disk. If provided, the
        function will load the file in blocks and concatenate them to minimise
        memory spikes for very large recordings.
    """

    audio_path = Path(path)
    _validate_path(audio_path)

    data_iter: Iterable[np.ndarray]
    if chunk_size is None:
        samples, sr = sf.read(audio_path, always_2d=True, dtype="float32")
        data_iter = (samples,)
    else:
        with sf.SoundFile(audio_path, "r") as f:
            sr = f.samplerate
            frames = []
            while True:
                block = f.read(chunk_size, dtype="float32", always_2d=True)
                if block.size == 0:
                    break
                frames.append(block)
            data_iter = frames
        if not data_iter:
            raise AudioLoadingError(f"Audio file '{audio_path}' is empty.")

    stacked = np.vstack(tuple(data_iter))
    channels = stacked.shape[1]
    if mono:
        samples = _to_mono(stacked).astype(np.float32)
        channel_count = 1
    else:
        samples = stacked.T.astype(np.float32)  # shape: (channels, samples)
        channel_count = channels

    samples = _remove_dc_offset(samples)
    samples = _normalise(samples)
    samples, sr = _resample_if_needed(samples, sr, target_sr)

    if dtype != np.float32:
        samples = samples.astype(dtype, copy=False)

    sample_length = samples.shape[-1] if samples.ndim > 1 else len(samples)
    duration = float(sample_length / sr) if sr > 0 else 0.0
    return AudioSignal(samples=samples, sample_rate=sr, channels=channel_count, duration=duration, path=audio_path)
