"""Tests para CsvCollector — escritura a fichero CSV."""
import sys
import os
import csv
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.core.types import FrameRecord, ExerciseSnapshot
from src.data_collection.csv_collector import CsvCollector
from src.utils.body_angles import ANGLE_NAMES
from src.utils.velocity_tracker import VELOCITY_NAMES


# Total: 6 identity + 4 bbox + 4 final_labels + 7 raw+meta + 6 segment_meta +
#        3 quality + 19 angles + 5 velocity + 51 keypoints = 105
_EXPECTED_COLS = 105


def _make_frame_record(frame_number: int = 0) -> FrameRecord:
    return FrameRecord(
        timestamp=1710000000.0 + frame_number,
        frame_number=frame_number,
        track_id=1,
        fps=30.0,
        session_id="test-session",
        video_id="0",
        bbox=(10.0, 20.0, 110.0, 220.0),
        activity_label="exercise",
        exercise_label="squat",
        phase_label="down",
        rep_id=0,
        raw_activity_label="squat",
        raw_exercise_label="squat",
        raw_phase_label="down",
        raw_confidence=0.85,
        decision_status="confirmed",
        event_id=1,
        label_source="immediate",
        mean_kpt_conf=0.8,
        visible_kpt_count=14,
        is_valid_pose=True,
        body_angles={name: 90.0 for name in ANGLE_NAMES},
        velocity={name: 0.0 for name in VELOCITY_NAMES},
        keypoints_xy=np.random.rand(17, 2).astype(np.float32),
        keypoints_conf=np.random.rand(17).astype(np.float32),
        keypoints_xy_norm=np.random.rand(17, 2).astype(np.float32),
        exercises={"squat": ExerciseSnapshot(name="squat", phase="down")},
    )


class TestCsvCollector:
    def test_creates_file_with_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = CsvCollector(output_dir=Path(tmpdir), filename_prefix="test_frames")
            collector.on_frame(_make_frame_record())
            collector.close()

            filepath = Path(tmpdir) / "test_frames.csv"
            assert filepath.exists()

            with open(filepath) as f:
                reader = csv.reader(f)
                header = next(reader)
                assert header[0] == "session_id"
                assert "kp0_x_norm" in header
                assert "elbow_left_deg" in header
                assert len(header) == _EXPECTED_COLS

    def test_row_count_matches_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = CsvCollector(output_dir=Path(tmpdir), filename_prefix="test_frames")

            for i in range(5):
                collector.on_frame(_make_frame_record(frame_number=i))
            collector.close()

            filepath = Path(tmpdir) / "test_frames.csv"
            with open(filepath) as f:
                reader = csv.reader(f)
                rows = list(reader)
                assert len(rows) == 6  # 1 header + 5 data rows

    def test_flush_still_readable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = CsvCollector(output_dir=Path(tmpdir), filename_prefix="test_frames")
            collector.on_frame(_make_frame_record())
            collector.flush()

            filepath = Path(tmpdir) / "test_frames.csv"
            assert filepath.exists()
            assert filepath.stat().st_size > 0

            collector.close()
