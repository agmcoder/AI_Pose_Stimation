"""Tests for keypoint_normalizer — bbox-based normalization."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.utils.keypoint_normalizer import normalize_keypoints


class TestNormalizeKeypoints:
    def test_center_of_bbox_is_half(self):
        """A keypoint at the center of the bbox should normalize to (0.5, 0.5)."""
        coords = np.array([[150.0, 250.0]] * 17, dtype=np.float32)
        bbox = (100.0, 200.0, 200.0, 300.0)  # 100x100 bbox
        result = normalize_keypoints(coords, bbox)
        np.testing.assert_allclose(result[0], [0.5, 0.5], atol=1e-6)

    def test_top_left_is_zero(self):
        """A keypoint at bbox origin should normalize to (0, 0)."""
        coords = np.array([[100.0, 200.0]] * 17, dtype=np.float32)
        bbox = (100.0, 200.0, 300.0, 400.0)
        result = normalize_keypoints(coords, bbox)
        np.testing.assert_allclose(result[0], [0.0, 0.0], atol=1e-6)

    def test_bottom_right_is_one(self):
        """A keypoint at bbox bottom-right should normalize to (1, 1)."""
        coords = np.array([[300.0, 400.0]] * 17, dtype=np.float32)
        bbox = (100.0, 200.0, 300.0, 400.0)
        result = normalize_keypoints(coords, bbox)
        np.testing.assert_allclose(result[0], [1.0, 1.0], atol=1e-6)

    def test_outside_bbox_exceeds_range(self):
        """Keypoints outside bbox may produce values > 1 or < 0."""
        coords = np.array([[350.0, 450.0]] * 17, dtype=np.float32)
        bbox = (100.0, 200.0, 300.0, 400.0)
        result = normalize_keypoints(coords, bbox)
        assert result[0, 0] > 1.0
        assert result[0, 1] > 1.0

    def test_zero_area_bbox_returns_zeros(self):
        """Degenerate bbox should return zeros to avoid division by zero."""
        coords = np.random.rand(17, 2).astype(np.float32) * 100
        bbox = (100.0, 200.0, 100.0, 200.0)  # zero area
        result = normalize_keypoints(coords, bbox)
        np.testing.assert_array_equal(result, np.zeros((17, 2)))

    def test_output_shape(self):
        """Output shape should match input shape."""
        coords = np.random.rand(17, 2).astype(np.float32)
        bbox = (0.0, 0.0, 100.0, 100.0)
        result = normalize_keypoints(coords, bbox)
        assert result.shape == (17, 2)
