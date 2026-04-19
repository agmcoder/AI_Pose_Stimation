"""
src/utils/keypoint_normalizer.py

Bounding-box normalization for COCO keypoints.

Strategy
--------
    kp_x_norm = (kp_x - bbox_x1) / (bbox_x2 - bbox_x1)
    kp_y_norm = (kp_y - bbox_y1) / (bbox_y2 - bbox_y1)

This produces translation- and scale-invariant coordinates in ~[0, 1]
(keypoints slightly outside the bbox may exceed this range).

Chosen over frame-size normalization because the project already provides
a per-person bounding box, making this the most natural approach for
per-person ML features.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def normalize_keypoints(
    coords: np.ndarray,
    bbox: Tuple[float, float, float, float],
) -> np.ndarray:
    """
    Normalize keypoint coordinates relative to the bounding box.

    Parameters
    ----------
    coords : np.ndarray, shape (17, 2)
        Raw pixel coordinates (x, y) per keypoint.
    bbox : (x1, y1, x2, y2)
        Person bounding box in pixel coordinates.

    Returns
    -------
    np.ndarray, shape (17, 2)
        Normalized coordinates.  Returns zeros if bbox has zero area.
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    if w < 1e-6 or h < 1e-6:
        return np.zeros_like(coords)

    normalized = np.empty_like(coords)
    normalized[:, 0] = (coords[:, 0] - x1) / w
    normalized[:, 1] = (coords[:, 1] - y1) / h
    return normalized
