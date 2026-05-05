"""Tests para FrameRecord, RepRecord y ExerciseSnapshot."""
import numpy as np
import json
import sys
import os

# Asegurar que src está en el path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.types import (
    ExerciseSnapshot,
    FrameRecord,
    RepRecord,
)
from src.utils.body_angles import ANGLE_NAMES
from src.utils.velocity_tracker import VELOCITY_NAMES


# Total columns: 6 identity + 4 bbox + 4 final_labels + 7 raw+meta +
#                6 segment_meta + 3 quality + 19 angles + 5 velocity +
#                51 keypoints = 105
_EXPECTED_COLS = 105


def _make_snapshot(**overrides) -> ExerciseSnapshot:
    defaults = dict(name="squat", phase="down", angles={"knee": 95.0, "hip": 62.0},
                    feedback="", valid_down=True)
    defaults.update(overrides)
    return ExerciseSnapshot(**defaults)


def _make_frame_record(**overrides) -> FrameRecord:
    defaults = dict(
        timestamp=1710000000.0,
        frame_number=42,
        track_id=7,
        fps=30.0,
        session_id="2026-04-07_12-00-00",
        video_id="0",
        bbox=(100.0, 200.0, 300.0, 400.0),
        activity_label="exercise",
        exercise_label="squat",
        phase_label="down",
        rep_id=3,
        raw_activity_label="squat",
        raw_exercise_label="squat",
        raw_phase_label="down",
        raw_confidence=0.92,
        decision_status="confirmed",
        event_id=1,
        label_source="immediate",
        mean_kpt_conf=0.85,
        visible_kpt_count=15,
        is_valid_pose=True,
        body_angles={name: 90.0 for name in ANGLE_NAMES},
        velocity={name: 0.01 for name in VELOCITY_NAMES},
        keypoints_xy=np.random.rand(17, 2).astype(np.float32),
        keypoints_conf=np.random.rand(17).astype(np.float32),
        keypoints_xy_norm=np.random.rand(17, 2).astype(np.float32),
        exercises={"squat": _make_snapshot()},
    )
    defaults.update(overrides)
    return FrameRecord(**defaults)


def _make_rep_record(**overrides) -> RepRecord:
    defaults = dict(
        timestamp=1710000005.0,
        track_id=7,
        exercise_name="squat",
        rep_number=3,
        was_valid=True,
        metrics={"min_knee_angle": 85.0, "min_hip_angle": 62.0},
        duration_frames=45,
        feedback="✅ Rep completada",
    )
    defaults.update(overrides)
    return RepRecord(**defaults)


# ── ExerciseSnapshot ─────────────────────────────────────────────────────────

class TestExerciseSnapshot:
    def test_to_dict_contains_all_fields(self):
        snap = _make_snapshot()
        d = snap.to_dict()
        assert d["name"] == "squat"
        assert d["phase"] == "down"
        assert d["angles"] == {"knee": 95.0, "hip": 62.0}
        assert d["valid_down"] is True

    def test_to_dict_is_json_serializable(self):
        snap = _make_snapshot()
        s = json.dumps(snap.to_dict())
        assert isinstance(s, str)


# ── FrameRecord ──────────────────────────────────────────────────────────────

class TestFrameRecord:
    def test_to_dict_is_json_serializable(self):
        record = _make_frame_record()
        d = record.to_dict()
        s = json.dumps(d)
        assert isinstance(s, str)

    def test_to_dict_roundtrip(self):
        record = _make_frame_record()
        d = record.to_dict()
        parsed = json.loads(json.dumps(d))
        assert parsed["frame_number"] == 42
        assert parsed["track_id"] == 7
        assert parsed["session_id"] == "2026-04-07_12-00-00"
        assert parsed["exercise_label"] == "squat"
        assert parsed["phase_label"] == "down"
        assert parsed["rep_id"] == 3
        assert parsed["is_valid_pose"] is True
        assert len(parsed["keypoints_xy"]) == 17
        assert len(parsed["keypoints_conf"]) == 17
        assert len(parsed["keypoints_xy_norm"]) == 17
        assert "squat" in parsed["exercises"]

    def test_to_csv_row_length(self):
        record = _make_frame_record()
        row = record.to_csv_row()
        assert len(row) == _EXPECTED_COLS

    def test_csv_header_length_matches_row(self):
        record = _make_frame_record()
        header = FrameRecord.csv_header()
        row = record.to_csv_row()
        assert len(header) == _EXPECTED_COLS
        assert len(header) == len(row)

    def test_csv_header_contains_new_columns(self):
        header = FrameRecord.csv_header()
        assert "session_id" in header
        assert "video_id" in header
        assert "activity_label" in header
        assert "exercise_label" in header
        assert "phase_label" in header
        assert "rep_id" in header
        assert "mean_kpt_conf" in header
        assert "visible_kpt_count" in header
        assert "is_valid_pose" in header
        assert "elbow_left_deg" in header
        assert "trunk_deg" in header
        assert "hip_center_vy" in header
        assert "kp0_x_norm" in header
        assert "kp16_conf" in header
        # New deferred-label columns
        assert "raw_activity_label" in header
        assert "raw_exercise_label" in header
        assert "raw_phase_label" in header
        assert "raw_confidence" in header
        assert "decision_status" in header
        assert "event_id" in header
        assert "label_source" in header
        # Temporal segmentation columns
        assert "segment_id" in header
        assert "segment_start_frame" in header
        assert "segment_end_frame" in header
        assert "segment_start_time" in header
        assert "segment_end_time" in header
        assert "prediction_confidence" in header

    def test_csv_row_order_identity_first(self):
        record = _make_frame_record()
        row = record.to_csv_row()
        # First 6 should be identity fields
        assert row[0] == "2026-04-07_12-00-00"  # session_id
        assert row[1] == "0"                     # video_id
        assert row[2] == 1710000000.0            # timestamp
        assert row[3] == 42                      # frame_number
        assert row[4] == 7                       # track_id
        assert row[5] == 30.0                    # fps

    def test_csv_row_angles_present(self):
        record = _make_frame_record()
        row = record.to_csv_row()
        header = FrameRecord.csv_header()
        idx = header.index("elbow_left_deg")
        assert row[idx] == 90.0

    def test_csv_row_none_angles(self):
        """Angles can be None when pose quality is insufficient."""
        record = _make_frame_record(body_angles={name: None for name in ANGLE_NAMES})
        row = record.to_csv_row()
        header = FrameRecord.csv_header()
        idx = header.index("elbow_left_deg")
        assert row[idx] is None

    def test_to_dict_exercises_multi(self):
        """Verifica soporte multi-ejercicio."""
        record = _make_frame_record(exercises={
            "squat": _make_snapshot(name="squat"),
            "jumping_jack": _make_snapshot(name="jumping_jack", phase="up",
                                           angles={"arm_spread": 170.0}),
        })
        d = record.to_dict()
        assert "squat" in d["exercises"]
        assert "jumping_jack" in d["exercises"]
        assert d["exercises"]["jumping_jack"]["angles"]["arm_spread"] == 170.0


# ── RepRecord ────────────────────────────────────────────────────────────────

class TestRepRecord:
    def test_to_dict_is_json_serializable(self):
        record = _make_rep_record()
        s = json.dumps(record.to_dict())
        assert isinstance(s, str)

    def test_to_dict_contains_generic_metrics(self):
        record = _make_rep_record(
            exercise_name="jumping_jack",
            metrics={"arm_spread": 170.0, "leg_spread": 45.0},
        )
        d = record.to_dict()
        assert d["exercise_name"] == "jumping_jack"
        assert d["metrics"]["arm_spread"] == 170.0
        assert d["metrics"]["leg_spread"] == 45.0
