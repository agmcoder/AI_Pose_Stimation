"""Tests para DeferredLabelBuffer — etiquetado diferido con buffer temporal."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.core.types import FrameRecord
from src.data_collection.deferred_label_buffer import DeferredLabelBuffer


def _make_record(
    frame_number: int,
    track_id: int = 1,
    raw_exercise: str = "none",
    raw_phase: str = "none",
    raw_confidence: float = 0.0,
) -> FrameRecord:
    """Helper: create a minimal FrameRecord with raw labels."""
    raw_activity = raw_exercise if raw_exercise != "none" else "none"
    return FrameRecord(
        timestamp=1710000000.0 + frame_number * 0.033,
        frame_number=frame_number,
        track_id=track_id,
        fps=30.0,
        session_id="test",
        video_id="0",
        raw_activity_label=raw_activity,
        raw_exercise_label=raw_exercise,
        raw_phase_label=raw_phase,
        raw_confidence=raw_confidence,
        # Final labels start as "pending" — buffer fills them
        activity_label="pending",
        exercise_label="pending",
        phase_label="pending",
        decision_status="pending",
    )


# ── Scenario 1: Confirmed exercise with retroactive labeling ─────────────


class TestConfirmedExercise:
    """Push a sequence where squat is confirmed after confirm_window frames."""

    def test_retroactive_labeling(self):
        buf = DeferredLabelBuffer(confirm_window=5, cooldown_window=3)
        all_flushed: list[FrameRecord] = []

        # 3 frames of no_exercise, then 20 frames of squat, then 5 of none
        for i in range(3):
            all_flushed.extend(buf.push(_make_record(i, raw_exercise="none")))
        for i in range(3, 23):
            all_flushed.extend(buf.push(_make_record(i, raw_exercise="squat", raw_phase="down")))
        for i in range(23, 28):
            all_flushed.extend(buf.push(_make_record(i, raw_exercise="none")))

        # Flush remaining
        all_flushed.extend(buf.flush_all())

        # Sort for inspection
        all_flushed.sort(key=lambda r: r.frame_number)

        # All 28 frames should be flushed
        assert len(all_flushed) == 28, f"Expected 28, got {len(all_flushed)}"

        # No duplicates
        frame_numbers = [r.frame_number for r in all_flushed]
        assert len(set(frame_numbers)) == 28

        # Frames 0-2 should be no_exercise
        for r in all_flushed[:3]:
            assert r.activity_label == "no_exercise", f"Frame {r.frame_number}: {r.activity_label}"
            assert r.decision_status == "confirmed"

        # Frames 3-22 should be exercise (squat)
        exercise_frames = [r for r in all_flushed if r.activity_label == "exercise"]
        assert len(exercise_frames) >= 15, f"Expected >=15 exercise frames, got {len(exercise_frames)}"

        # All exercise frames should have event_id > 0
        for r in exercise_frames:
            assert r.event_id > 0, f"Frame {r.frame_number} has event_id=0"
            assert r.exercise_label == "squat"

        # No frames should have "pending" as final label
        for r in all_flushed:
            assert r.activity_label != "pending", f"Frame {r.frame_number} still pending"
            assert r.decision_status != "pending", f"Frame {r.frame_number} still pending status"

    def test_event_id_is_consistent(self):
        buf = DeferredLabelBuffer(confirm_window=5, cooldown_window=3)
        all_flushed: list[FrameRecord] = []

        for i in range(20):
            all_flushed.extend(buf.push(_make_record(i, raw_exercise="squat")))
        for i in range(20, 25):
            all_flushed.extend(buf.push(_make_record(i, raw_exercise="none")))
        all_flushed.extend(buf.flush_all())
        all_flushed.sort(key=lambda r: r.frame_number)

        exercise_frames = [r for r in all_flushed if r.activity_label == "exercise"]
        if exercise_frames:
            event_ids = {r.event_id for r in exercise_frames}
            assert len(event_ids) == 1, f"Expected 1 event_id, got {event_ids}"


# ── Scenario 2: False positive (candidate not confirmed) ─────────────────


class TestFalsePositive:
    """Exercise signal appears briefly but doesn't reach confirm_window."""

    def test_short_burst_flushed_as_no_exercise(self):
        buf = DeferredLabelBuffer(confirm_window=10, cooldown_window=3)
        all_flushed: list[FrameRecord] = []

        # 5 frames of no_exercise
        for i in range(5):
            all_flushed.extend(buf.push(_make_record(i)))

        # 7 frames of squat (less than confirm_window=10)
        for i in range(5, 12):
            all_flushed.extend(buf.push(_make_record(i, raw_exercise="squat")))

        # Signal drops
        for i in range(12, 17):
            all_flushed.extend(buf.push(_make_record(i)))

        all_flushed.extend(buf.flush_all())
        all_flushed.sort(key=lambda r: r.frame_number)

        assert len(all_flushed) == 17

        # All should be no_exercise since squat was never confirmed
        for r in all_flushed:
            assert r.activity_label == "no_exercise", (
                f"Frame {r.frame_number}: expected no_exercise, got {r.activity_label}"
            )


# ── Scenario 3: Clean transition none → squat → none ────────────────────


class TestTransition:
    """Validate no gaps or duplicates across a full transition."""

    def test_no_gaps_no_duplicates(self):
        buf = DeferredLabelBuffer(confirm_window=5, cooldown_window=3)
        all_flushed: list[FrameRecord] = []

        # 10 frames none, 20 frames squat, 10 frames none
        for i in range(10):
            all_flushed.extend(buf.push(_make_record(i)))
        for i in range(10, 30):
            all_flushed.extend(buf.push(_make_record(i, raw_exercise="squat")))
        for i in range(30, 40):
            all_flushed.extend(buf.push(_make_record(i)))

        all_flushed.extend(buf.flush_all())
        all_flushed.sort(key=lambda r: r.frame_number)

        # All 40 frames present
        frame_numbers = [r.frame_number for r in all_flushed]
        assert frame_numbers == list(range(40)), f"Missing or duplicate frames: {frame_numbers}"

        # No pending labels
        for r in all_flushed:
            assert r.activity_label != "pending"

    def test_transition_labels_are_correct(self):
        buf = DeferredLabelBuffer(confirm_window=5, cooldown_window=3)
        all_flushed: list[FrameRecord] = []

        for i in range(5):
            all_flushed.extend(buf.push(_make_record(i)))
        for i in range(5, 25):
            all_flushed.extend(buf.push(_make_record(i, raw_exercise="squat")))
        for i in range(25, 35):
            all_flushed.extend(buf.push(_make_record(i)))

        all_flushed.extend(buf.flush_all())
        all_flushed.sort(key=lambda r: r.frame_number)

        # First 5 should be no_exercise
        for r in all_flushed[:5]:
            assert r.activity_label == "no_exercise"

        # Some middle frames should be exercise
        exercise_count = sum(1 for r in all_flushed if r.activity_label == "exercise")
        assert exercise_count > 0, "No exercise frames found"

        # Last frames should be no_exercise
        for r in all_flushed[-3:]:
            assert r.activity_label == "no_exercise"


# ── Scenario 4: Flush on stream end ──────────────────────────────────────


class TestFlushOnEnd:
    """Active exercise when stream ends — should flush gracefully."""

    def test_active_exercise_flushed_on_end(self):
        buf = DeferredLabelBuffer(confirm_window=5, cooldown_window=3)
        all_flushed: list[FrameRecord] = []

        # 20 frames of squat — never followed by no_exercise
        for i in range(20):
            all_flushed.extend(buf.push(_make_record(i, raw_exercise="squat")))

        # Stream ends
        all_flushed.extend(buf.flush_all())
        all_flushed.sort(key=lambda r: r.frame_number)

        assert len(all_flushed) == 20
        # Should all be exercise (confirmed after confirm_window)
        exercise_count = sum(1 for r in all_flushed if r.activity_label == "exercise")
        assert exercise_count >= 15, f"Expected >=15 exercise, got {exercise_count}"

    def test_candidate_flushed_as_no_exercise_on_end(self):
        buf = DeferredLabelBuffer(confirm_window=10, cooldown_window=3)
        all_flushed: list[FrameRecord] = []

        # 7 frames of squat — below confirm_window=10
        for i in range(7):
            all_flushed.extend(buf.push(_make_record(i, raw_exercise="squat")))

        all_flushed.extend(buf.flush_all())
        all_flushed.sort(key=lambda r: r.frame_number)

        assert len(all_flushed) == 7
        for r in all_flushed:
            assert r.activity_label == "no_exercise"


# ── Scenario 5: Multi-track independence ─────────────────────────────────


class TestMultiTrack:
    """Two track_ids should have independent state machines."""

    def test_independent_tracks(self):
        buf = DeferredLabelBuffer(confirm_window=5, cooldown_window=3)
        all_flushed: list[FrameRecord] = []

        # Track 1: doing squat. Track 2: idle.
        for i in range(20):
            all_flushed.extend(
                buf.push(_make_record(i, track_id=1, raw_exercise="squat"))
            )
            all_flushed.extend(
                buf.push(_make_record(i, track_id=2, raw_exercise="none"))
            )

        # End both
        for i in range(20, 25):
            all_flushed.extend(buf.push(_make_record(i, track_id=1)))
            all_flushed.extend(buf.push(_make_record(i, track_id=2)))

        all_flushed.extend(buf.flush_all())

        track1 = sorted([r for r in all_flushed if r.track_id == 1],
                         key=lambda r: r.frame_number)
        track2 = sorted([r for r in all_flushed if r.track_id == 2],
                         key=lambda r: r.frame_number)

        # Track 1 should have exercise frames
        t1_exercise = sum(1 for r in track1 if r.activity_label == "exercise")
        assert t1_exercise > 0, "Track 1 should have exercise frames"

        # Track 2 should be entirely no_exercise
        for r in track2:
            assert r.activity_label == "no_exercise", (
                f"Track 2 frame {r.frame_number}: {r.activity_label}"
            )

        # Both tracks have all 25 frames
        assert len(track1) == 25
        assert len(track2) == 25


# ── Scenario 6: Cooldown recovery ────────────────────────────────────────


class TestCooldownRecovery:
    """Brief detection drop during exercise should not split the event."""

    def test_brief_drop_recovers(self):
        buf = DeferredLabelBuffer(confirm_window=5, cooldown_window=5)
        all_flushed: list[FrameRecord] = []

        # 10 frames squat, 3 frames drop (within cooldown=5), 10 frames squat
        for i in range(10):
            all_flushed.extend(buf.push(_make_record(i, raw_exercise="squat")))
        for i in range(10, 13):
            all_flushed.extend(buf.push(_make_record(i)))
        for i in range(13, 23):
            all_flushed.extend(buf.push(_make_record(i, raw_exercise="squat")))

        # End
        for i in range(23, 30):
            all_flushed.extend(buf.push(_make_record(i)))

        all_flushed.extend(buf.flush_all())
        all_flushed.sort(key=lambda r: r.frame_number)

        assert len(all_flushed) == 30

        # Should have a single event_id for the whole exercise block
        exercise_frames = [r for r in all_flushed if r.activity_label == "exercise"]
        if exercise_frames:
            event_ids = {r.event_id for r in exercise_frames}
            assert len(event_ids) == 1, f"Expected single event, got {event_ids}"


# ── Scenario 7: Duplicate frame guard ────────────────────────────────────


class TestDuplicateGuard:
    """Pushing the same frame_number twice should be silently skipped."""

    def test_duplicate_frame_skipped(self):
        buf = DeferredLabelBuffer(confirm_window=5, cooldown_window=3)
        all_flushed: list[FrameRecord] = []

        for i in range(5):
            all_flushed.extend(buf.push(_make_record(i)))

        # Push duplicate of frame 3 (already flushed)
        dup_results = buf.push(_make_record(3))
        all_flushed.extend(dup_results)

        all_flushed.extend(buf.flush_all())

        frame_numbers = [r.frame_number for r in all_flushed]
        # Should not have duplicate frame 3
        assert frame_numbers.count(3) <= 1


# ── Scenario 8: Confidence-aware transitions (new) ──────────────────────


class TestConfidenceAware:
    """Test hysteresis-based transitions using raw_confidence."""

    def test_high_confidence_confirms_segment(self):
        """Frames with high LSTM confidence should produce exercise labels."""
        buf = DeferredLabelBuffer(
            confirm_window=5, cooldown_window=3,
            enter_threshold=0.5, exit_threshold=0.3,
        )
        all_flushed: list[FrameRecord] = []

        # 5 idle frames
        for i in range(5):
            all_flushed.extend(buf.push(_make_record(i, raw_confidence=0.1)))

        # 15 high-confidence squat frames
        for i in range(5, 20):
            all_flushed.extend(buf.push(_make_record(
                i, raw_exercise="squat", raw_confidence=0.85, raw_phase="down"
            )))

        # 5 idle frames to close
        for i in range(20, 25):
            all_flushed.extend(buf.push(_make_record(i, raw_confidence=0.1)))

        all_flushed.extend(buf.flush_all())
        all_flushed.sort(key=lambda r: r.frame_number)

        assert len(all_flushed) == 25

        exercise_frames = [r for r in all_flushed if r.activity_label == "exercise"]
        assert len(exercise_frames) >= 10, (
            f"Expected >=10 exercise frames, got {len(exercise_frames)}"
        )

        # Exercise frames should have segment metadata
        for r in exercise_frames:
            assert r.segment_id > 0
            assert r.segment_start_frame >= 0
            assert r.segment_end_frame >= r.segment_start_frame
            assert r.prediction_confidence > 0

    def test_low_confidence_rejected(self):
        """Frames below enter_threshold should not create exercise segments."""
        buf = DeferredLabelBuffer(
            confirm_window=5, cooldown_window=3,
            enter_threshold=0.5, exit_threshold=0.3,
        )
        all_flushed: list[FrameRecord] = []

        # 20 frames with low confidence (below enter_threshold=0.5)
        for i in range(20):
            all_flushed.extend(buf.push(_make_record(
                i, raw_exercise="squat", raw_confidence=0.35
            )))

        all_flushed.extend(buf.flush_all())
        all_flushed.sort(key=lambda r: r.frame_number)

        # All should be no_exercise since confidence never reaches enter_threshold
        for r in all_flushed:
            assert r.activity_label == "no_exercise", (
                f"Frame {r.frame_number}: expected no_exercise, got {r.activity_label}"
            )


# ── Scenario 9: Segment metadata (new) ──────────────────────────────────


class TestSegmentMetadata:
    """Validate segment_id, boundaries, and confidence are set correctly."""

    def test_segment_boundaries_correct(self):
        buf = DeferredLabelBuffer(
            confirm_window=5, cooldown_window=3,
            min_segment_frames=5,
        )
        all_flushed: list[FrameRecord] = []

        # 5 idle, 15 squat, 5 idle
        for i in range(5):
            all_flushed.extend(buf.push(_make_record(i)))
        for i in range(5, 20):
            all_flushed.extend(buf.push(_make_record(
                i, raw_exercise="squat", raw_phase="down"
            )))
        for i in range(20, 25):
            all_flushed.extend(buf.push(_make_record(i)))

        all_flushed.extend(buf.flush_all())
        all_flushed.sort(key=lambda r: r.frame_number)

        exercise_frames = [r for r in all_flushed if r.activity_label == "exercise"]
        if exercise_frames:
            # All exercise frames should share the same segment_id
            segment_ids = {r.segment_id for r in exercise_frames}
            assert len(segment_ids) == 1, f"Expected 1 segment_id, got {segment_ids}"

            # All should share the same start/end frame
            starts = {r.segment_start_frame for r in exercise_frames}
            ends = {r.segment_end_frame for r in exercise_frames}
            assert len(starts) == 1
            assert len(ends) == 1

        # Non-exercise frames should have segment_id = 0
        idle_frames = [r for r in all_flushed if r.activity_label == "no_exercise"]
        for r in idle_frames:
            assert r.segment_id == 0, f"Frame {r.frame_number}: unexpected segment_id={r.segment_id}"

    def test_min_segment_rejects_short(self):
        """Segments shorter than min_segment_frames should be rejected."""
        buf = DeferredLabelBuffer(
            confirm_window=3, cooldown_window=3,
            min_segment_frames=10,
        )
        all_flushed: list[FrameRecord] = []

        # 3 idle frames
        for i in range(3):
            all_flushed.extend(buf.push(_make_record(i)))

        # 5 squat frames — enough for confirm_window=3 but < min_segment=10
        for i in range(3, 8):
            all_flushed.extend(buf.push(_make_record(i, raw_exercise="squat")))

        # End
        for i in range(8, 15):
            all_flushed.extend(buf.push(_make_record(i)))

        all_flushed.extend(buf.flush_all())
        all_flushed.sort(key=lambda r: r.frame_number)

        # All should be no_exercise since segment was too short
        for r in all_flushed:
            assert r.activity_label == "no_exercise", (
                f"Frame {r.frame_number}: expected no_exercise, got {r.activity_label}"
            )


# ── Scenario 10: Rep counting within segments (new) ─────────────────────


class TestRepCounting:
    """Rep IDs should be assigned based on phase transitions within segments."""

    def test_rep_id_from_phase_transitions(self):
        buf = DeferredLabelBuffer(confirm_window=3, cooldown_window=3)
        all_flushed: list[FrameRecord] = []

        # 3 idle frames
        for i in range(3):
            all_flushed.extend(buf.push(_make_record(i)))

        # Squat segment with 2 reps: down→up→down→up
        phases = (
            ["down"] * 5 + ["up"] * 5 +   # rep 1
            ["down"] * 5 + ["up"] * 5      # rep 2
        )
        for i, phase in enumerate(phases, start=3):
            all_flushed.extend(buf.push(_make_record(
                i, raw_exercise="squat", raw_phase=phase,
            )))

        # End segment
        for i in range(23, 28):
            all_flushed.extend(buf.push(_make_record(i)))

        all_flushed.extend(buf.flush_all())
        all_flushed.sort(key=lambda r: r.frame_number)

        exercise_frames = [r for r in all_flushed if r.activity_label == "exercise"]
        if exercise_frames:
            max_rep = max(r.rep_id for r in exercise_frames)
            assert max_rep == 2, f"Expected 2 reps, got {max_rep}"

        # Idle frames should have rep_id = 0
        idle_frames = [r for r in all_flushed if r.activity_label == "no_exercise"]
        for r in idle_frames:
            assert r.rep_id == 0
