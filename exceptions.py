"""Custom exceptions for the FrequenCipher package."""
from __future__ import annotations


class FrequenCipherError(Exception):
    """Base exception for all FrequenCipher errors."""


class AudioLoadingError(FrequenCipherError):
    """Raised when an audio file cannot be loaded or validated."""


class UnsupportedFormatError(AudioLoadingError):
    """Raised when attempting to load an unsupported audio format."""


class AnalysisError(FrequenCipherError):
    """Raised when an analysis step fails unexpectedly."""
