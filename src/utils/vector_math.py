"""
src/utils/vector_math.py

Geometric helpers for COCO keypoint-based exercise detection.

All functions operate on numpy arrays and are stateless.
Angles are computed via dot-product (numerically stable for 0-180°).
"""
from __future__ import annotations

import numpy as np


def angle_at_vertex(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Angle (degrees) at vertex B formed by rays BA and BC.

    Uses dot-product: θ = arccos( (BA · BC) / (|BA| |BC|) )
    Stable for the 0°–180° range required by joint-angle calculations.

    Parameters
    ----------
    a, b, c : np.ndarray, shape (2,)
        2-D coordinates.  *b* is the vertex.

    Returns
    -------
    float
        Angle in degrees, clamped to [0, 180].
    """
    ba = a - b
    bc = c - b
    norm = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
    cos_val = np.dot(ba, bc) / norm
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))


def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Midpoint between two 2-D points."""
    return (a + b) * 0.5


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two 2-D points."""
    return float(np.linalg.norm(a - b))


def torso_length(coords: np.ndarray, scores: np.ndarray, min_conf: float) -> float | None:
    """
    Average distance from shoulder to hip (left + right).

    Returns None if any of the four keypoints is below *min_conf*.

    Indices — COCO:
      5: left_shoulder   6: right_shoulder
     11: left_hip       12: right_hip
    """
    idxs = (5, 6, 11, 12)
    if scores[list(idxs)].min() < min_conf:
        return None
    left  = euclidean(coords[5], coords[11])
    right = euclidean(coords[6], coords[12])
    return (left + right) * 0.5


def hip_width(coords: np.ndarray, scores: np.ndarray, min_conf: float) -> float | None:
    """
    Horizontal distance between left and right hip.

    Returns None if either hip is below *min_conf*.

    Indices — COCO: 11: left_hip  12: right_hip
    """
    if scores[11] < min_conf or scores[12] < min_conf:
        return None
    return abs(float(coords[11][0] - coords[12][0]))
