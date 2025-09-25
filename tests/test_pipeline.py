from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from frequencipher.analysis import run_full_analysis
from frequencipher.ingestion import load_audio


@pytest.fixture()
def tmp_audio_file(tmp_path: Path) -> Path:
    sr = 22050
    t = np.linspace(0, 2, sr * 2, endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)
    mod = 0.2 * np.sin(2 * np.pi * 5 * t)
    stereo = np.stack([tone + mod, tone - mod], axis=1)
    path = tmp_path / "test.wav"
    sf.write(path, stereo, sr)
    return path


def test_load_audio_returns_audio_signal(tmp_audio_file: Path) -> None:
    audio = load_audio(tmp_audio_file, target_sr=44100, mono=True)
    assert audio.sample_rate == 44100
    assert audio.channels == 1
    assert audio.samples.ndim == 1
    assert audio.duration > 1.9


def test_run_full_analysis_structure(tmp_audio_file: Path) -> None:
    audio, results = run_full_analysis(str(tmp_audio_file), target_sr=22050)
    assert audio.sample_rate == 22050
    assert "spectral" in results
    assert "summaries" in results["spectral"]
    mfcc_summary = results["spectral"]["summaries"]["mfcc"]
    assert "mean" in mfcc_summary
    assert isinstance(mfcc_summary["mean"], float)
    assert "phase" in results
    assert "backmask" in results
    assert "subliminal" in results
    assert "steganography" in results
    assert "temporal" in results
    assert "watermark" in results
    assert "anomaly" in results
