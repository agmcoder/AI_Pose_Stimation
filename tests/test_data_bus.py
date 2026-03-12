"""Tests para DataCollectionBus — fan-out a múltiples collectors."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.core.types import FrameRecord, RepRecord, ExerciseSnapshot
from src.data_collection.bus import DataCollectionBus
from src.data_collection.in_memory_collector import InMemoryCollector


def _make_frame_record() -> FrameRecord:
    return FrameRecord(
        timestamp=1710000000.0,
        frame_number=1,
        track_id=1,
        bbox=(0.0, 0.0, 100.0, 200.0),
        keypoints_xy=np.zeros((17, 2), dtype=np.float32),
        keypoints_conf=np.ones(17, dtype=np.float32),
        exercises={"squat": ExerciseSnapshot(name="squat", phase="down")},
    )


def _make_rep_record() -> RepRecord:
    return RepRecord(
        timestamp=1710000005.0,
        track_id=1,
        exercise_name="squat",
        rep_number=1,
        was_valid=True,
        metrics={"min_knee_angle": 90.0},
    )


class TestDataCollectionBus:
    def test_fan_out_on_frame(self):
        c1 = InMemoryCollector()
        c2 = InMemoryCollector()
        bus = DataCollectionBus([c1, c2])

        record = _make_frame_record()
        bus.on_frame(record)

        assert len(c1.frames) == 1
        assert len(c2.frames) == 1
        assert c1.frames[0] is record

    def test_fan_out_on_rep(self):
        c1 = InMemoryCollector()
        c2 = InMemoryCollector()
        bus = DataCollectionBus([c1, c2])

        record = _make_rep_record()
        bus.on_rep(record)

        assert len(c1.reps) == 1
        assert len(c2.reps) == 1

    def test_empty_bus_no_crash(self):
        bus = DataCollectionBus([])
        bus.on_frame(_make_frame_record())
        bus.on_rep(_make_rep_record())
        bus.flush()
        bus.close()

    def test_close_delegates(self):
        c1 = InMemoryCollector()
        bus = DataCollectionBus([c1])
        bus.on_frame(_make_frame_record())
        bus.close()  # should not raise

    def test_add_collector(self):
        bus = DataCollectionBus()
        c1 = InMemoryCollector()
        bus.add(c1)
        bus.on_frame(_make_frame_record())
        assert len(c1.frames) == 1
