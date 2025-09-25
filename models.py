"""Data models used across the FrequenCipher toolkit."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Optional

import numpy as np


@dataclass(slots=True)
class AudioSignal:
    """Container for audio samples and associated metadata."""

    samples: np.ndarray
    sample_rate: int
    channels: int
    duration: float
    path: Optional[Path] = None

    def to_dict(self) -> Dict[str, object]:
        """Serialize the metadata (excluding raw samples) into a dictionary."""

        metadata = asdict(self)
        # Replace samples with lightweight descriptors to avoid dumping raw arrays
        metadata["samples"] = {
            "shape": tuple(self.samples.shape),
            "dtype": str(self.samples.dtype),
        }
        return metadata


@dataclass(slots=True)
class SummaryStatistics:
    """Numeric summary for a feature vector or matrix."""

    mean: float
    std: float
    median: float
    min: float
    max: float
    percentile_25: float
    percentile_75: float

    def to_dict(self) -> Dict[str, float]:
        """Return a JSON-serialisable representation of the statistics."""

        return asdict(self)


SummaryMapping = Mapping[str, SummaryStatistics]
MutableSummaryMapping = MutableMapping[str, SummaryStatistics]
