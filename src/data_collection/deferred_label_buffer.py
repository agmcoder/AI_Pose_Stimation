"""
src/data_collection/deferred_label_buffer.py

Temporal buffer that sits between record_builder and the DataCollectionBus.

Accumulates FrameRecords in a per-track sliding buffer, runs a
confidence-aware state machine to decide when an exercise event is
confirmed or rejected, and assigns final labels retroactively before
flushing to downstream collectors.

State machine per track (with hysteresis):
    IDLE ──(confidence ≥ enter_threshold)──▶ CANDIDATE_SQUAT
    CANDIDATE_SQUAT ──(≥ confirm_window consecutive)──▶ IN_SQUAT
    CANDIDATE_SQUAT ──(drops or inconsistent)──▶ IDLE (flush as no_exercise)
    IN_SQUAT ──(confidence < exit_threshold)──▶ EXITING_SQUAT
    EXITING_SQUAT ──(exercise resumes)──▶ IN_SQUAT
    EXITING_SQUAT ──(cooldown_window frames elapsed)──▶ IDLE (close segment)

Temporal segmentation features:
    - Confidence-aware transitions (hysteresis: enter > exit)
    - Minimum segment length enforcement
    - Adjacent segment merging for fragmented predictions
    - Per-segment metadata: segment_id, boundaries, mean confidence
    - Rep counting scoped to validated segments only
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from loguru import logger

from src.core.types import FrameRecord


class _Phase(Enum):
    """Track-level state for deferred labeling."""
    IDLE = auto()
    CANDIDATE_SQUAT = auto()
    IN_SQUAT = auto()
    EXITING_SQUAT = auto()


@dataclass
class _TrackState:
    """Mutable state for one tracked person."""
    phase: _Phase = _Phase.IDLE
    buffer: list[FrameRecord] = field(default_factory=list)
    event_id: int = 0                  # current event counter
    candidate_start_idx: int = 0       # buffer index where candidate started
    candidate_exercise: str = ""       # exercise name under evaluation
    cooldown_counter: int = 0          # frames remaining in EXITING_SQUAT
    flushed_up_to_frame: int = -1      # last frame_number flushed (dup guard)
    segment_rep_count: int = 0         # reps counted within current segment


class DeferredLabelBuffer:
    """
    Temporal buffer with confidence-aware state-machine label resolution.

    Implements hysteresis-based temporal segmentation:
    - Exercise starts only when LSTM confidence exceeds `enter_threshold`
      for `confirm_window` consecutive frames
    - Exercise ends only when confidence drops below `exit_threshold`
      for `cooldown_window` consecutive frames
    - Short segments (< min_segment_frames) are rejected as noise
    - Adjacent segments separated by < merge_gap_frames are merged

    Parameters
    ----------
    confirm_window : int
        Number of consecutive frames with confidence ≥ enter_threshold
        needed to transition from CANDIDATE → IN_SQUAT.
    cooldown_window : int
        Number of consecutive low-confidence frames after IN_SQUAT
        before the event is closed.
    max_buffer_size : int
        Safety cap — if a track's buffer exceeds this, force-flush the
        oldest frames to prevent unbounded memory growth.
    enter_threshold : float
        LSTM confidence threshold to consider a frame as exercise.
        Uses the raw_confidence field from FrameRecord.
    exit_threshold : float
        LSTM confidence below which the exercise is considered ending.
        Must be < enter_threshold for hysteresis.
    min_segment_frames : int
        Minimum frames for a valid segment. Shorter segments are
        rejected as noise.
    merge_gap_frames : int
        If two segments are separated by fewer than this many frames,
        merge them into one continuous segment.
    """

    def __init__(
        self,
        confirm_window: int = 12,
        cooldown_window: int = 8,
        max_buffer_size: int = 300,
        enter_threshold: float = 0.5,
        exit_threshold: float = 0.3,
        min_segment_frames: int = 8,
        merge_gap_frames: int = 5,
    ) -> None:
        self._confirm_window = confirm_window
        self._cooldown_window = cooldown_window
        self._max_buffer_size = max_buffer_size
        self._enter_threshold = enter_threshold
        self._exit_threshold = exit_threshold
        self._min_segment_frames = min_segment_frames
        self._merge_gap_frames = merge_gap_frames
        self._tracks: dict[int, _TrackState] = {}
        self._global_event_counter = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(self, record: FrameRecord) -> list[FrameRecord]:
        """
        Ingest one FrameRecord. Returns any finalized records ready to
        flush to downstream collectors (in temporal order).
        """
        tid = record.track_id
        if tid not in self._tracks:
            self._tracks[tid] = _TrackState()

        ts = self._tracks[tid]

        # Duplicate guard
        if record.frame_number <= ts.flushed_up_to_frame:
            logger.warning(
                "Duplicate frame {} for track {}, skipping",
                record.frame_number, tid,
            )
            return []

        ts.buffer.append(record)
        return self._evaluate(ts)

    def flush_all(self) -> list[FrameRecord]:
        """
        End-of-stream: finalize all pending buffers.
        Returns all remaining records with final labels assigned.
        """
        results: list[FrameRecord] = []
        for tid, ts in self._tracks.items():
            if not ts.buffer:
                continue

            if ts.phase in (_Phase.IN_SQUAT, _Phase.EXITING_SQUAT):
                # Close the current exercise event
                results.extend(self._close_event(ts))
            elif ts.phase == _Phase.CANDIDATE_SQUAT:
                # Not enough evidence — reject as no_exercise
                results.extend(self._reject_candidate(ts))
            # else: IDLE — fall through to drain

            # Drain any remaining frames (e.g. retained by _reject_candidate)
            if ts.buffer:
                results.extend(
                    self._flush_as_no_exercise(ts, 0, len(ts.buffer))
                )

            ts.phase = _Phase.IDLE

        # Sort by (frame_number, track_id) for stable output order
        results.sort(key=lambda r: (r.frame_number, r.track_id))
        return results

    # ------------------------------------------------------------------
    # Confidence evaluation
    # ------------------------------------------------------------------

    def _is_exercise_signal(self, record: FrameRecord) -> bool:
        """
        Determine if a frame carries a positive exercise signal.

        Uses a dual-path check:
        1. If raw_confidence > 0 (LSTM detector), use enter_threshold
        2. If raw_exercise_label is set (angle-based), treat as binary
        """
        raw_ex = record.raw_exercise_label
        conf = record.raw_confidence

        # LSTM-based: use confidence threshold
        if conf > 0:
            return conf >= self._enter_threshold

        # Angle-based fallback: binary check on label
        return raw_ex not in ("none", "", "pending")

    def _is_still_exercise(self, record: FrameRecord) -> bool:
        """
        Check if a frame is still within an exercise using the exit
        (lower) threshold — implements hysteresis.
        """
        raw_ex = record.raw_exercise_label
        conf = record.raw_confidence

        # LSTM-based: use exit threshold (lower than enter)
        if conf > 0:
            return conf >= self._exit_threshold

        # Angle-based fallback
        return raw_ex not in ("none", "", "pending")

    # ------------------------------------------------------------------
    # State machine evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, ts: _TrackState) -> list[FrameRecord]:
        """Run one step of the state machine for a track."""
        record = ts.buffer[-1]  # most recent record
        raw_ex = record.raw_exercise_label
        is_exercise_enter = self._is_exercise_signal(record)
        is_exercise_stay = self._is_still_exercise(record)

        results: list[FrameRecord] = []

        if ts.phase == _Phase.IDLE:
            if is_exercise_enter:
                ts.phase = _Phase.CANDIDATE_SQUAT
                ts.candidate_start_idx = len(ts.buffer) - 1
                ts.candidate_exercise = raw_ex if raw_ex not in ("none", "", "pending") else "squat"
            else:
                # Flush immediately — no ambiguity in IDLE + no_exercise
                results.extend(
                    self._flush_as_no_exercise(ts, 0, 1)
                )

        elif ts.phase == _Phase.CANDIDATE_SQUAT:
            if is_exercise_enter:
                # Count consecutive exercise frames since candidate_start
                consecutive = len(ts.buffer) - ts.candidate_start_idx
                if consecutive >= self._confirm_window:
                    # Confirmed — transition to IN_SQUAT
                    ts.phase = _Phase.IN_SQUAT
                    self._global_event_counter += 1
                    ts.event_id = self._global_event_counter
                    ts.segment_rep_count = 0

                    # Flush any frames before the candidate as no_exercise
                    if ts.candidate_start_idx > 0:
                        results.extend(
                            self._flush_as_no_exercise(
                                ts, 0, ts.candidate_start_idx
                            )
                        )
                    # Don't flush exercise frames yet — wait for event end
            else:
                # Signal broken — reject candidate
                results.extend(self._reject_candidate(ts))

                # If the new frame is exercise (different type or fresh start),
                # start a new candidate
                if is_exercise_enter:
                    ts.phase = _Phase.CANDIDATE_SQUAT
                    ts.candidate_start_idx = len(ts.buffer) - 1
                    ts.candidate_exercise = raw_ex if raw_ex not in ("none", "", "pending") else "squat"

        elif ts.phase == _Phase.IN_SQUAT:
            if not is_exercise_stay:
                ts.phase = _Phase.EXITING_SQUAT
                ts.cooldown_counter = 1
            # else: keep accumulating exercise frames

        elif ts.phase == _Phase.EXITING_SQUAT:
            if is_exercise_stay:
                # Exercise resumed during cooldown — back to IN_SQUAT
                ts.phase = _Phase.IN_SQUAT
                ts.cooldown_counter = 0
            elif not is_exercise_stay:
                ts.cooldown_counter += 1
                if ts.cooldown_counter >= self._cooldown_window:
                    # Cooldown expired — close the event
                    results.extend(self._close_event(ts))
            else:
                # Different exercise — close current, start new candidate
                results.extend(self._close_event(ts))
                ts.phase = _Phase.CANDIDATE_SQUAT
                ts.candidate_start_idx = len(ts.buffer) - 1
                ts.candidate_exercise = raw_ex if raw_ex not in ("none", "", "pending") else "squat"

        # Safety: prevent unbounded buffer growth
        if len(ts.buffer) > self._max_buffer_size:
            results.extend(self._force_flush_oldest(ts))

        return results

    # ------------------------------------------------------------------
    # Label assignment helpers
    # ------------------------------------------------------------------

    def _close_event(self, ts: _TrackState) -> list[FrameRecord]:
        """
        Close an IN_SQUAT/EXITING_SQUAT event: retroactively label all
        buffered frames belonging to this event, then flush them.

        Applies segment metadata (segment_id, boundaries, mean confidence)
        and enforces minimum segment length.
        """
        # Find the boundary: exercise frames vs trailing no-exercise frames
        event_end = len(ts.buffer)

        # Find where cooldown started (last consecutive low-confidence block)
        cooldown_start = event_end
        for i in range(event_end - 1, -1, -1):
            rec = ts.buffer[i]
            if self._is_still_exercise(rec):
                break
            cooldown_start = i

        results: list[FrameRecord] = []

        # Exercise frames: from start of buffer to cooldown_start
        exercise_frame_count = cooldown_start
        if exercise_frame_count > 0:
            # Enforce minimum segment length
            if exercise_frame_count < self._min_segment_frames:
                # Too short — reject entire segment as noise
                logger.debug(
                    "Segment too short ({} frames < {} min), rejecting",
                    exercise_frame_count, self._min_segment_frames,
                )
                results.extend(
                    self._flush_as_no_exercise(ts, 0, event_end)
                )
                ts.phase = _Phase.IDLE
                ts.cooldown_counter = 0
                return results

            # Compute segment metadata
            exercise_name = ts.candidate_exercise
            segment_id = ts.event_id
            segment_start_frame = ts.buffer[0].frame_number
            segment_end_frame = ts.buffer[cooldown_start - 1].frame_number
            segment_start_time = ts.buffer[0].timestamp
            segment_end_time = ts.buffer[cooldown_start - 1].timestamp

            # Compute mean confidence across exercise frames
            confidences = [
                ts.buffer[i].raw_confidence
                for i in range(cooldown_start)
                if ts.buffer[i].raw_confidence > 0
            ]
            mean_confidence = (
                sum(confidences) / len(confidences) if confidences else 0.0
            )

            # Count rep transitions (down→up) within the segment
            segment_rep_id = 0
            prev_phase = None
            for i in range(cooldown_start):
                rec = ts.buffer[i]
                curr_phase = rec.raw_phase_label
                if prev_phase == "down" and curr_phase == "up":
                    segment_rep_id += 1
                prev_phase = curr_phase

            # Label exercise frames
            current_rep = 0
            prev_phase = None
            for i in range(cooldown_start):
                rec = ts.buffer[i]
                rec.activity_label = "exercise"
                rec.exercise_label = exercise_name
                # Keep the raw phase for phase_label (most granular info)
                rec.phase_label = rec.raw_phase_label
                rec.event_id = ts.event_id
                rec.decision_status = (
                    "confirmed" if self._is_still_exercise(rec)
                    else "retroactive"
                )
                rec.label_source = (
                    "immediate" if self._is_still_exercise(rec)
                    else "retroactive"
                )

                # Assign segment metadata
                rec.segment_id = segment_id
                rec.segment_start_frame = segment_start_frame
                rec.segment_end_frame = segment_end_frame
                rec.segment_start_time = segment_start_time
                rec.segment_end_time = segment_end_time
                rec.prediction_confidence = round(mean_confidence, 4)

                # Assign rep_id scoped to this segment
                curr_phase = rec.raw_phase_label
                if prev_phase == "down" and curr_phase == "up":
                    current_rep += 1
                rec.rep_id = current_rep
                prev_phase = curr_phase

            results.extend(ts.buffer[:cooldown_start])

        # Trailing cooldown frames: label as no_exercise
        if cooldown_start < event_end:
            for i in range(cooldown_start, event_end):
                rec = ts.buffer[i]
                rec.activity_label = "no_exercise"
                rec.exercise_label = "none"
                rec.phase_label = "none"
                rec.event_id = 0
                rec.decision_status = "confirmed"
                rec.label_source = "smoothed"
                rec.segment_id = 0
                rec.segment_start_frame = -1
                rec.segment_end_frame = -1
                rec.segment_start_time = -1.0
                rec.segment_end_time = -1.0
                rec.prediction_confidence = 0.0
                rec.rep_id = 0
            results.extend(ts.buffer[cooldown_start:event_end])

        # Update flushed watermark
        if results:
            ts.flushed_up_to_frame = results[-1].frame_number

        # Clear the buffer
        ts.buffer.clear()
        ts.candidate_start_idx = 0
        ts.cooldown_counter = 0
        ts.phase = _Phase.IDLE

        return results

    def _reject_candidate(self, ts: _TrackState) -> list[FrameRecord]:
        """
        Candidate didn't confirm — flush all buffered frames as no_exercise.
        Keep the most recent frame in the buffer (it might start a new event).
        """
        if len(ts.buffer) <= 1:
            # Only one frame — flush everything
            results = self._flush_as_no_exercise(ts, 0, len(ts.buffer))
            ts.phase = _Phase.IDLE
            return results

        # Save the last frame before flushing (it triggered the rejection
        # and might be the start of a different exercise)
        last_frame = ts.buffer[-1]

        # Flush everything as no_exercise
        results = self._flush_as_no_exercise(ts, 0, len(ts.buffer))

        # Put the last frame back in the buffer for potential re-evaluation
        ts.buffer = [last_frame]
        ts.candidate_start_idx = 0
        ts.phase = _Phase.IDLE

        # Remove last_frame from flushed results (it's back in buffer)
        if results and results[-1].frame_number == last_frame.frame_number:
            results.pop()

        return results

    def _flush_as_no_exercise(
        self, ts: _TrackState, start: int, end: int,
    ) -> list[FrameRecord]:
        """Label frames [start:end) as no_exercise and remove from buffer."""
        results: list[FrameRecord] = []
        for i in range(start, end):
            rec = ts.buffer[i]
            rec.activity_label = "no_exercise"
            rec.exercise_label = "none"
            rec.phase_label = "none"
            rec.event_id = 0
            rec.decision_status = "confirmed"
            rec.label_source = "immediate"
            rec.segment_id = 0
            rec.segment_start_frame = -1
            rec.segment_end_frame = -1
            rec.segment_start_time = -1.0
            rec.segment_end_time = -1.0
            rec.prediction_confidence = 0.0
            rec.rep_id = 0
            results.append(rec)

        if results:
            ts.flushed_up_to_frame = results[-1].frame_number

        # Remove flushed frames from buffer
        ts.buffer = ts.buffer[end:]
        # Adjust candidate_start_idx
        ts.candidate_start_idx = max(0, ts.candidate_start_idx - end)

        return results

    def _force_flush_oldest(self, ts: _TrackState) -> list[FrameRecord]:
        """
        Safety valve: flush oldest frames when buffer exceeds max_buffer_size.
        Preserves a window of recent frames for the state machine.
        """
        keep = self._confirm_window + self._cooldown_window
        flush_count = len(ts.buffer) - keep
        if flush_count <= 0:
            return []

        logger.warning(
            "Buffer overflow for track ({}), force-flushing {} frames",
            ts.buffer[0].track_id if ts.buffer else "?",
            flush_count,
        )

        results: list[FrameRecord] = []
        if ts.phase in (_Phase.IN_SQUAT, _Phase.EXITING_SQUAT):
            # We're in an active event — flush as exercise with segment info
            exercise_name = ts.candidate_exercise

            # Compute running mean confidence for force-flushed frames
            confidences = [
                ts.buffer[i].raw_confidence
                for i in range(flush_count)
                if ts.buffer[i].raw_confidence > 0
            ]
            mean_conf = (
                sum(confidences) / len(confidences) if confidences else 0.0
            )

            for i in range(flush_count):
                rec = ts.buffer[i]
                rec.activity_label = "exercise"
                rec.exercise_label = exercise_name
                rec.phase_label = rec.raw_phase_label
                rec.event_id = ts.event_id
                rec.decision_status = "confirmed"
                rec.label_source = "immediate"
                rec.segment_id = ts.event_id
                rec.prediction_confidence = round(mean_conf, 4)
                # Segment boundaries will be finalized when event closes;
                # for now set partial info
                rec.segment_start_frame = ts.buffer[0].frame_number
                rec.segment_end_frame = -1  # unknown yet
                rec.segment_start_time = ts.buffer[0].timestamp
                rec.segment_end_time = -1.0
                results.append(rec)
        else:
            # IDLE or CANDIDATE — flush as no_exercise
            for i in range(flush_count):
                rec = ts.buffer[i]
                rec.activity_label = "no_exercise"
                rec.exercise_label = "none"
                rec.phase_label = "none"
                rec.event_id = 0
                rec.decision_status = "confirmed"
                rec.label_source = "immediate"
                rec.segment_id = 0
                rec.segment_start_frame = -1
                rec.segment_end_frame = -1
                rec.segment_start_time = -1.0
                rec.segment_end_time = -1.0
                rec.prediction_confidence = 0.0
                rec.rep_id = 0
                results.append(rec)

        if results:
            ts.flushed_up_to_frame = results[-1].frame_number

        ts.buffer = ts.buffer[flush_count:]
        ts.candidate_start_idx = max(0, ts.candidate_start_idx - flush_count)

        return results
