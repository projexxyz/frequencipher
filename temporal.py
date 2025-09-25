"""Temporal manipulation and authenticity checks."""
from __future__ import annotations

from typing import Dict

import librosa
import numpy as np

from .statistics import summarise_array


def check_temporal_manipulation(samples: np.ndarray, sample_rate: int) -> Dict[str, float]:
    """Detect tempo or pitch manipulation via beat and zero-crossing analysis."""

    if samples.ndim > 1:
        samples = np.mean(samples, axis=0)

    zcr = librosa.feature.zero_crossing_rate(samples)[0]
    zcr_summary = summarise_array(zcr).to_dict()

    onset_env = librosa.onset.onset_strength(y=samples, sr=sample_rate)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sample_rate)
    tempo_value = float(np.atleast_1d(tempo)[0]) if np.size(tempo) else float("nan")
    beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate)
    if beat_times.size > 1:
        beat_intervals = np.diff(beat_times)
        tempo_stability = summarise_array(beat_intervals).to_dict()
    else:
        nan = float("nan")
        tempo_stability = {
            "mean": nan,
            "std": nan,
            "median": nan,
            "min": nan,
            "max": nan,
            "percentile_25": nan,
            "percentile_75": nan,
        }

    return {
        "tempo_bpm": tempo_value,
        "tempo_interval_std": tempo_stability["std"],
        "tempo_interval_percentile_75": tempo_stability["percentile_75"],
        "zero_crossing_rate_mean": zcr_summary["mean"],
        "zero_crossing_rate_std": zcr_summary["std"],
    }
