"""FrequenCipher audio forensics toolkit."""

from .analysis import run_full_analysis
from .ingestion import load_audio
from .spectral import compute_spectral_features
from .phase import detect_phase_anomalies
from .backmask import detect_backmasking
from .subliminal import detect_subliminal
from .anomaly import score_anomalies
from .steganography import detect_steganography
from .watermark import detect_watermark
from .temporal import check_temporal_manipulation
from .report import generate_report
from .models import AudioSignal

__all__ = [
    "run_full_analysis",
    "load_audio",
    "compute_spectral_features",
    "detect_phase_anomalies",
    "detect_backmasking",
    "detect_subliminal",
    "score_anomalies",
    "detect_steganography",
    "detect_watermark",
    "check_temporal_manipulation",
    "generate_report",
    "AudioSignal",
]
