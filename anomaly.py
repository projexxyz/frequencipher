"""
Machine learning-based anomaly detection.
"""
import numpy as np
from typing import Dict

try:
    # Import IsolationForest if available
    from sklearn.ensemble import IsolationForest  # type: ignore
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


def score_anomalies(features: np.ndarray) -> Dict[str, float]:
    """
    Score anomalies using either an Isolation Forest or a statistical z‑score.

    Parameters
    ----------
    features : np.ndarray
        2D feature matrix where rows correspond to observations and columns to
        features extracted from the audio.

    Returns
    -------
    dict
        A dictionary with a single key ``anomaly_score``. Higher scores
        indicate more anomalous data.
    """
    if _HAS_SKLEARN and features.ndim == 2:
        # Use IsolationForest to compute anomaly scores
        clf = IsolationForest(n_estimators=100, contamination='auto', random_state=0)
        clf.fit(features)
        # The lower (more negative) the score, the more anomalous the observation
        scores = -clf.score_samples(features)
        return {"anomaly_score": float(np.mean(scores))}
    else:
        # Fall back to z-score on flattened features
        flattened = features.flatten()
        z_scores = np.abs((flattened - flattened.mean()) / (flattened.std() + 1e-8))
        return {"anomaly_score": float(np.mean(z_scores))}
