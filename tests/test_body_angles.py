"""Tests for body_angles — canonical angle computation."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.utils.body_angles import compute_body_angles, ANGLE_NAMES


def _standing_pose() -> tuple[np.ndarray, np.ndarray]:
    """
    Approximate standing pose with all keypoints visible.
    Coordinates in image space (y increases downward).
    """
    coords = np.array([
        [300, 100],   # 0  nose
        [310, 95],    # 1  left_eye
        [290, 95],    # 2  right_eye
        [320, 100],   # 3  left_ear
        [280, 100],   # 4  right_ear
        [340, 200],   # 5  left_shoulder
        [260, 200],   # 6  right_shoulder
        [360, 300],   # 7  left_elbow
        [240, 300],   # 8  right_elbow
        [370, 400],   # 9  left_wrist
        [230, 400],   # 10 right_wrist
        [330, 400],   # 11 left_hip
        [270, 400],   # 12 right_hip
        [330, 550],   # 13 left_knee
        [270, 550],   # 14 right_knee
        [330, 700],   # 15 left_ankle
        [270, 700],   # 16 right_ankle
    ], dtype=np.float32)
    scores = np.full(17, 0.9, dtype=np.float32)
    return coords, scores


class TestBodyAngles:
    def test_all_angles_present(self):
        """All 19 angle names should be keys in the result."""
        coords, scores = _standing_pose()
        angles = compute_body_angles(coords, scores)
        assert set(angles.keys()) == set(ANGLE_NAMES)

    def test_standing_pose_produces_values(self):
        """For a high-confidence standing pose, all angles should be non-None."""
        coords, scores = _standing_pose()
        angles = compute_body_angles(coords, scores)
        for name, val in angles.items():
            assert val is not None, f"{name} should not be None for a visible standing pose"
            assert 0.0 <= val <= 180.0, f"{name}={val} out of range"

    def test_low_confidence_returns_none(self):
        """When all keypoints have low confidence, angles should be None."""
        coords, _ = _standing_pose()
        scores = np.full(17, 0.1, dtype=np.float32)
        angles = compute_body_angles(coords, scores, min_conf=0.3)
        for name, val in angles.items():
            assert val is None, f"{name} should be None for low-confidence pose"

    def test_partial_confidence(self):
        """Only angles with sufficient keypoints should be computed."""
        coords, scores = _standing_pose()
        # Zero out right side confidence
        scores[6] = 0.1   # right_shoulder
        scores[8] = 0.1   # right_elbow
        scores[10] = 0.1  # right_wrist
        scores[12] = 0.1  # right_hip
        scores[14] = 0.1  # right_knee
        scores[16] = 0.1  # right_ankle

        angles = compute_body_angles(coords, scores, min_conf=0.3)
        # Left-side angles should still be computed
        assert angles["elbow_left_deg"] is not None
        assert angles["knee_left_deg"] is not None
        # Right-side angles should be None
        assert angles["elbow_right_deg"] is None
        assert angles["knee_right_deg"] is None

    def test_knee_angle_range_for_standing(self):
        """Standing pose should have near-straight knees (≈ 170-180°)."""
        coords, scores = _standing_pose()
        angles = compute_body_angles(coords, scores)
        assert angles["knee_left_deg"] is not None
        assert angles["knee_left_deg"] > 150.0  # nearly straight

    def test_returns_rounded_values(self):
        """All non-None values should be rounded to 2 decimal places."""
        coords, scores = _standing_pose()
        angles = compute_body_angles(coords, scores)
        for name, val in angles.items():
            if val is not None:
                # Check that it has at most 2 decimal places
                assert val == round(val, 2), f"{name}={val} not rounded to 2 decimals"
