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


def _make_frame_record(frame_number: int = 0) -> FrameRecord:
    return FrameRecord(
        timestamp=1710000000.0 + frame_number,
        frame_number=frame_number,
        track_id=1,
        bbox=(10.0, 20.0, 110.0, 220.0),
        keypoints_xy=np.random.rand(17, 2).astype(np.float32),
        keypoints_conf=np.random.rand(17).astype(np.float32),
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
                assert header[0] == "timestamp"
                assert "kp0_x" in header
                assert len(header) == 58  # 3 meta + 4 bbox + 17*3 kps

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
