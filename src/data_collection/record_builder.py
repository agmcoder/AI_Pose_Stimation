"""
src/data_collection/record_builder.py

Transforms the domain model (Person) into the persistence model (FrameRecord).

All derived features (quality, normalization, angles, velocity) are computed
here — the single funnel between the domain pipeline and the data collectors.
"""
from __future__ import annotations

from typing import Optional

from src.core.types import Person, ExerciseSnapshot, FrameRecord
from src.utils.body_angles import compute_body_angles
from src.utils.keypoint_normalizer import normalize_keypoints
from src.utils.pose_quality import compute_pose_quality
from src.utils.velocity_tracker import VelocityTracker


def build_frame_records(
    persons: list[Person],
    frame_number: int,
    timestamp: float,
    current_fps: float = 0.0,
    session_id: str = "",
    video_id: str = "",
    velocity_tracker: Optional[VelocityTracker] = None,
) -> list[FrameRecord]:
    """
    Transforma la lista de Person del frame actual en FrameRecords.

    Función pura (excepto VelocityTracker que mantiene estado por track).
    Mantiene la lógica de transformación fuera de main.py (SRP).
    """
    records: list[FrameRecord] = []

    for person in persons:
        if person.keypoints is None:
            continue

        coords = person.keypoints.coords
        scores = person.keypoints.scores

        # ── 1. Quality metrics ───────────────────────────────────────────
        quality = compute_pose_quality(scores)

        # ── 2. Normalize keypoints ───────────────────────────────────────
        kp_norm = normalize_keypoints(coords, person.bbox)

        # ── 3. Body angles (only if pose is valid) ───────────────────────
        body_angles = {}
        if quality.is_valid_pose:
            body_angles = compute_body_angles(coords, scores)

        # ── 4. Velocity ──────────────────────────────────────────────────
        velocity = {}
        if velocity_tracker is not None:
            velocity = velocity_tracker.update(
                person.track_id, coords, scores, person.bbox,
            )

        # ── 5. Labels from exercise state ──────────────────────────────
        raw_activity, raw_exercise, raw_phase, rep_id, raw_conf = (
            _extract_labels(person)
        )

        # ── 6. Exercise snapshots ────────────────────────────────────────
        snapshots = {
            ex_name: ExerciseSnapshot(
                name=ex_name,
                phase=state.phase,
                angles=dict(state.angles),
                feedback=state.feedback,
                valid_down=state.valid_down,
            )
            for ex_name, state in person.exercises.items()
        }

        records.append(FrameRecord(
            timestamp=timestamp,
            frame_number=frame_number,
            track_id=person.track_id,
            fps=current_fps,
            session_id=session_id,
            video_id=video_id,
            bbox=person.bbox,
            # Final labels — pending until DeferredLabelBuffer resolves
            activity_label="pending",
            exercise_label="pending",
            phase_label="pending",
            rep_id=rep_id,
            # Raw labels — instantaneous from detector
            raw_activity_label=raw_activity,
            raw_exercise_label=raw_exercise,
            raw_phase_label=raw_phase,
            raw_confidence=raw_conf,
            decision_status="pending",
            event_id=0,
            label_source="immediate",
            mean_kpt_conf=quality.mean_kpt_conf,
            visible_kpt_count=quality.visible_kpt_count,
            is_valid_pose=quality.is_valid_pose,
            body_angles=body_angles,
            velocity=velocity,
            keypoints_xy=coords.copy(),
            keypoints_conf=scores.copy(),
            keypoints_xy_norm=kp_norm,
            exercises=snapshots,
        ))

    return records


def _extract_labels(person: Person) -> tuple[str, str, str, int, float]:
    """
    Extract (activity_label, exercise_label, phase_label, rep_id, confidence)
    from active exercises.

    Conventions:
    - If no exercise is active → ("none", "none", "none", 0, 0.0)
    - If one exercise is active → that exercise's name/phase
    - If multiple → the first active exercise (alphabetically for stability)
    - rep_id comes from ExerciseState.rep_count
    - confidence: LSTM's probability (stored in ExerciseState.angle) or
      1.0 for angle-based detectors when in an active phase
    """
    if not person.exercises:
        return "none", "none", "none", 0, 0.0

    # Sort for deterministic ordering when multiple exercises active
    names = sorted(person.exercises.keys())
    primary = names[0]
    state = person.exercises[primary]

    # Phase label: pass through the detector's phase ("up"/"down") so that
    # downstream segmentation can use it for rep boundary detection.
    # Frames without a meaningful phase keep "none".
    phase_label = state.phase if state.phase in ("up", "down") else "none"

    # Extract confidence:
    # - For LSTM detectors, state.angle holds the model's probability
    # - For angle-based detectors, use 1.0 when in active phase, 0.0 otherwise
    confidence = float(state.angle) if state.angle is not None else 0.0

    return primary, primary, phase_label, state.rep_count, confidence
