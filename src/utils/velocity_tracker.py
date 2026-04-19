"""
src/utils/velocity_tracker.py

Per-track velocity features computed from frame-to-frame keypoint deltas.

Maintains a small per-track state (previous keypoints).  Thread-safe
because PipelineThread processes frames sequentially.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .vector_math import midpoint

# COCO indices
_LHI, _RHI = 11, 12
_LKN, _RKN = 13, 14

# Canonical velocity feature names
VELOCITY_NAMES: list[str] = [
    "hip_center_y_norm",
    "hip_center_vy",
    "knee_left_vy",
    "knee_right_vy",
    "mean_joint_speed",
]


@dataclass
class _TrackState:
    """Previous frame data for a tracked person."""
    coords: np.ndarray      # (17, 2)
    scores: np.ndarray      # (17,)
    bbox_h: float            # bbox height for normalization


class VelocityTracker:
    """
    Computes velocity features by comparing consecutive frames per track_id.

    Usage
    -----
    Called once per frame per person inside ``build_frame_records``.
    """

    def __init__(self, min_conf: float = 0.3) -> None:
        self._states: Dict[int, _TrackState] = {}
        self._min_conf = min_conf

    def update(
        self,
        track_id: int,
        coords: np.ndarray,
        scores: np.ndarray,
        bbox: Tuple[float, float, float, float],
    ) -> Dict[str, Optional[float]]:
        """
        Compute velocity features and update internal state.

        Returns dict with keys from ``VELOCITY_NAMES``.
        All values are ``None`` on the first frame for a given track.
        """
        x1, y1, x2, y2 = bbox
        bbox_h = max(y2 - y1, 1e-6)

        result: Dict[str, Optional[float]] = {name: None for name in VELOCITY_NAMES}

        # Hip center y (always available if hips visible)
        if scores[_LHI] >= self._min_conf and scores[_RHI] >= self._min_conf:
            hip_center = midpoint(coords[_LHI], coords[_RHI])
            result["hip_center_y_norm"] = round(float((hip_center[1] - y1) / bbox_h), 6)

        prev = self._states.get(track_id)

        if prev is not None:
            # Hip center velocity
            hip_ok_now = scores[_LHI] >= self._min_conf and scores[_RHI] >= self._min_conf
            hip_ok_prev = prev.scores[_LHI] >= self._min_conf and prev.scores[_RHI] >= self._min_conf

            if hip_ok_now and hip_ok_prev:
                hip_now = midpoint(coords[_LHI], coords[_RHI])
                hip_prev = midpoint(prev.coords[_LHI], prev.coords[_RHI])
                result["hip_center_vy"] = round(float(hip_now[1] - hip_prev[1]) / bbox_h, 6)

            # Knee velocities
            for name, idx in [("knee_left_vy", _LKN), ("knee_right_vy", _RKN)]:
                if scores[idx] >= self._min_conf and prev.scores[idx] >= self._min_conf:
                    result[name] = round(
                        float(coords[idx][1] - prev.coords[idx][1]) / bbox_h, 6
                    )

            # Mean joint speed — euclidean distance across all visible joints
            mask = (scores >= self._min_conf) & (prev.scores >= self._min_conf)
            if mask.sum() > 0:
                deltas = np.linalg.norm(coords[mask] - prev.coords[mask], axis=1)
                result["mean_joint_speed"] = round(float(np.mean(deltas)) / bbox_h, 6)

        # Store state for next frame
        self._states[track_id] = _TrackState(
            coords=coords.copy(),
            scores=scores.copy(),
            bbox_h=bbox_h,
        )

        return result

    def clear(self, track_id: int) -> None:
        """Remove state for a track that is no longer active."""
        self._states.pop(track_id, None)
