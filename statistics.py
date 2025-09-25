"""Utility helpers for computing descriptive statistics."""
from __future__ import annotations

from typing import Iterable

import numpy as np

from .models import SummaryStatistics


def _empty_statistics() -> SummaryStatistics:
    nan = float("nan")
    return SummaryStatistics(
        mean=nan,
        std=nan,
        median=nan,
        min=nan,
        max=nan,
        percentile_25=nan,
        percentile_75=nan,
    )


def summarise_array(values: np.ndarray | Iterable[float]) -> SummaryStatistics:
    """Compute robust summary statistics for an array-like object."""

    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    if array.size == 0:
        return _empty_statistics()

    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return _empty_statistics()

    return SummaryStatistics(
        mean=float(np.mean(finite)),
        std=float(np.std(finite)),
        median=float(np.median(finite)),
        min=float(np.min(finite)),
        max=float(np.max(finite)),
        percentile_25=float(np.percentile(finite, 25)),
        percentile_75=float(np.percentile(finite, 75)),
    )
