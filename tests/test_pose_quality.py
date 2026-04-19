"""Tests for pose_quality — quality metrics."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.utils.pose_quality import compute_pose_quality


class TestPoseQuality:
    def test_all_high_confidence(self):
        """All keypoints at high confidence → valid pose."""
        scores = np.full(17, 0.9, dtype=np.float32)
        q = compute_pose_quality(scores)
        assert q.mean_kpt_conf == pytest.approx(0.9, abs=0.01)
        assert q.visible_kpt_count == 17
        assert q.is_valid_pose is True

    def test_all_low_confidence(self):
        """All keypoints at low confidence → invalid pose."""
        scores = np.full(17, 0.1, dtype=np.float32)
        q = compute_pose_quality(scores)
        assert q.visible_kpt_count == 0
        assert q.is_valid_pose is False

    def test_partial_visibility(self):
        """Mix of high and low → check threshold behavior."""
        scores = np.zeros(17, dtype=np.float32)
        scores[:12] = 0.8  # 12 visible
        scores[12:] = 0.1  # 5 invisible
        q = compute_pose_quality(scores)
        assert q.visible_kpt_count == 12
        # mean = (12*0.8 + 5*0.1) / 17 ≈ 0.594
        assert q.mean_kpt_conf == pytest.approx(0.594, abs=0.01)
        assert q.is_valid_pose is True  # 12 ≥ 10 and mean ≥ 0.4

    def test_borderline_invalid(self):
        """Just below thresholds → invalid."""
        scores = np.zeros(17, dtype=np.float32)
        scores[:9] = 0.5   # 9 visible (< 10 min)
        scores[9:] = 0.1
        q = compute_pose_quality(scores)
        assert q.visible_kpt_count == 9
        assert q.is_valid_pose is False

    def test_custom_thresholds(self):
        """Custom thresholds override defaults."""
        scores = np.full(17, 0.35, dtype=np.float32)
        q = compute_pose_quality(scores, min_visible=5, min_mean_conf=0.3)
        assert q.is_valid_pose is True

    def test_frozen_dataclass(self):
        """PoseQuality should be immutable."""
        scores = np.full(17, 0.9, dtype=np.float32)
        q = compute_pose_quality(scores)
        with pytest.raises(AttributeError):
            q.mean_kpt_conf = 0.0
