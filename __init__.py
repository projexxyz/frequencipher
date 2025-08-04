"""FrequenCipher audio forensics toolkit."""

from .ingestion import load_audio
from .spectral import compute_spectrogram
from .phase import detect_phase_anomalies
from .backmask import detect_backmasking
from .subliminal import detect_subliminal
from .anomaly import score_anomalies
from .steganography import detect_steganography
from .watermark import detect_watermark
from .temporal import check_temporal_manipulation
from .report import generate_report
