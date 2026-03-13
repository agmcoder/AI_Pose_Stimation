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
        bbox=(100.0, 200.0, 300.0, 400.0),
        keypoints_xy=np.random.rand(17, 2).astype(np.float32),
        keypoints_conf=np.random.rand(17).astype(np.float32),
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
        assert len(parsed["keypoints_xy"]) == 17
        assert len(parsed["keypoints_conf"]) == 17
        assert "squat" in parsed["exercises"]
        assert parsed["exercises"]["squat"]["phase"] == "down"

    def test_to_csv_row_length(self):
        record = _make_frame_record()
        row = record.to_csv_row()
        # 4 metadata + 4 bbox + 17*3 keypoints = 59
        assert len(row) == 59

    def test_csv_header_length_matches_row(self):
        record = _make_frame_record()
        header = FrameRecord.csv_header()
        row = record.to_csv_row()
        assert len(header) == 59

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
