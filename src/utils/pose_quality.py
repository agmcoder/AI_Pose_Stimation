"""
src/utils/pose_quality.py

Per-frame pose quality metrics for data collection.

Stateless, operates on raw keypoint confidence arrays.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PoseQuality:
    """Quality metrics for a single person's pose detection."""
    mean_kpt_conf: float
    visible_kpt_count: int
    is_valid_pose: bool


def compute_pose_quality(
    scores: np.ndarray,
    *,
    visibility_threshold: float = 0.3,
    min_visible: int = 10,
    min_mean_conf: float = 0.4,
) -> PoseQuality:
    """
    Compute quality metrics from keypoint confidence scores.

    Parameters
    ----------
    scores : np.ndarray, shape (17,)
        Per-keypoint confidence scores.
    visibility_threshold : float
        Minimum confidence to consider a keypoint visible.
    min_visible : int
        Minimum visible keypoints for a valid pose.
    min_mean_conf : float
        Minimum mean confidence for a valid pose.

    Returns
    -------
    PoseQuality
        Frozen dataclass with mean_kpt_conf, visible_kpt_count, is_valid_pose.
    """
    mean_conf = float(np.mean(scores))
    visible = int(np.sum(scores >= visibility_threshold))
    valid = visible >= min_visible and mean_conf >= min_mean_conf

    return PoseQuality(
        mean_kpt_conf=round(mean_conf, 6),
        visible_kpt_count=visible,
        is_valid_pose=valid,
    )
