from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np

from src.utils.body_angles import ANGLE_NAMES
from src.utils.velocity_tracker import VELOCITY_NAMES

Frame = np.ndarray

@dataclass
class Keypoints:
    coords: np.ndarray   # shape (17, 2)
    scores: np.ndarray   # shape (17,)

@dataclass
class ExerciseState:
    name: str
    rep_count: int = 0
    phase: str = "up"
    angle: float = 0.0                              # ángulo principal (rodilla)
    angles: Dict[str, float] = field(default_factory=dict)  # ← nuevo: todas las articulaciones
    feedback: str = ""
    valid_down: bool = False                        # ← nuevo: bajada técnicamente válida

@dataclass
class Person:
    track_id: int
    bbox: Tuple[float, float, float, float]
    keypoints: Optional[Keypoints] = None
    exercises: Dict[str, ExerciseState] = field(default_factory=dict)


# ── Data Collection Structures ────────────────────────────────────────────────


@dataclass
class ExerciseSnapshot:
    """Estado de UN ejercicio en un frame concreto (exercise-agnostic)."""
    name: str
    phase: str
    angles: Dict[str, float] = field(default_factory=dict)
    feedback: str = ""
    valid_down: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "phase": self.phase,
            "angles": {k: float(v) for k, v in self.angles.items()},
            "feedback": self.feedback,
            "valid_down": self.valid_down,
        }


@dataclass
class FrameRecord:
    """
    Registro de UNA persona en UN frame.
    Captura keypoints raw + normalizados, ángulos canónicos, calidad,
    velocidad y estado de TODOS los ejercicios activos.
    """
    # ── Identity ──────────────────────────────────────────────────────────
    timestamp: float
    frame_number: int
    track_id: int
    fps: float
    session_id: str = ""
    video_id: str = ""

    # ── Detection ─────────────────────────────────────────────────────────
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    # ── Labels ────────────────────────────────────────────────────────────
    activity_label: str = "no_exercise"  # "exercise" | "no_exercise"
    exercise_label: str = "none"         # canonical name or "none"
    phase_label: str = "none"            # normalized phase or "none"
    rep_id: int = 0

    # ── Quality ───────────────────────────────────────────────────────────
    mean_kpt_conf: float = 0.0
    visible_kpt_count: int = 0
    is_valid_pose: bool = False

    # ── Body angles (19 canonical) ────────────────────────────────────────
    body_angles: Dict[str, Optional[float]] = field(default_factory=dict)

    # ── Velocity ──────────────────────────────────────────────────────────
    velocity: Dict[str, Optional[float]] = field(default_factory=dict)

    # ── Keypoints raw (kept for to_dict / JSONL) ──────────────────────────
    keypoints_xy: np.ndarray = field(default_factory=lambda: np.zeros((17, 2)))
    keypoints_conf: np.ndarray = field(default_factory=lambda: np.zeros(17))

    # ── Keypoints normalized ──────────────────────────────────────────────
    keypoints_xy_norm: np.ndarray = field(default_factory=lambda: np.zeros((17, 2)))

    # ── Exercises ─────────────────────────────────────────────────────────
    exercises: Dict[str, ExerciseSnapshot] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialización JSON-safe (todos los valores numpy → Python nativo)."""
        return {
            "timestamp": float(self.timestamp),
            "frame_number": int(self.frame_number),
            "track_id": int(self.track_id),
            "fps": float(self.fps),
            "session_id": self.session_id,
            "video_id": self.video_id,
            "bbox": [float(v) for v in self.bbox],
            "activity_label": self.activity_label,
            "exercise_label": self.exercise_label,
            "phase_label": self.phase_label,
            "rep_id": self.rep_id,
            "mean_kpt_conf": self.mean_kpt_conf,
            "visible_kpt_count": self.visible_kpt_count,
            "is_valid_pose": self.is_valid_pose,
            "body_angles": dict(self.body_angles),
            "velocity": dict(self.velocity),
            "keypoints_xy": self.keypoints_xy.tolist(),
            "keypoints_conf": self.keypoints_conf.tolist(),
            "keypoints_xy_norm": self.keypoints_xy_norm.tolist(),
            "exercises": {k: v.to_dict() for k, v in self.exercises.items()},
        }

    def to_csv_row(self) -> list:
        """
        Fila plana para CSV — orden canónico:
          identity (6) + bbox (4) + labels (4) + quality (3) +
          angles (19) + velocity (5) + keypoints_norm (51)
        = 92 columnas
        """
        row: list = [
            # ── Identity (6) ──
            self.session_id,
            self.video_id,
            self.timestamp,
            self.frame_number,
            self.track_id,
            self.fps,
            # ── Bbox (4) ──
            *self.bbox,
            # ── Labels (4) ──
            self.activity_label,
            self.exercise_label,
            self.phase_label,
            self.rep_id,
            # ── Quality (3) ──
            self.mean_kpt_conf,
            self.visible_kpt_count,
            self.is_valid_pose,
            # ── Angles (19) ──
            *[self.body_angles.get(name) for name in ANGLE_NAMES],
            # ── Velocity (5) ──
            *[self.velocity.get(name) for name in VELOCITY_NAMES],
        ]
        # ── Keypoints normalized (17 × 3 = 51) ──
        for i in range(17):
            row.extend([
                round(float(self.keypoints_xy_norm[i, 0]), 6),
                round(float(self.keypoints_xy_norm[i, 1]), 6),
                round(float(self.keypoints_conf[i]), 6),
            ])
        return row

    @staticmethod
    def csv_header() -> list[str]:
        """Nombres de columna para CSV."""
        header = [
            # Identity (6)
            "session_id", "video_id", "timestamp", "frame_number", "track_id", "fps",
            # Bbox (4)
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
            # Labels (4)
            "activity_label", "exercise_label", "phase_label", "rep_id",
            # Quality (3)
            "mean_kpt_conf", "visible_kpt_count", "is_valid_pose",
        ]
        # Angles (19)
        header.extend(ANGLE_NAMES)
        # Velocity (5)
        header.extend(VELOCITY_NAMES)
        # Keypoints normalized (51)
        for i in range(17):
            header.extend([f"kp{i}_x_norm", f"kp{i}_y_norm", f"kp{i}_conf"])
        return header


@dataclass
class RepRecord:
    """
    Registro de UNA repetición completada (exercise-agnostic).
    Cada detector publica sus métricas clave en el dict `metrics`.
    """
    timestamp: float
    track_id: int
    exercise_name: str
    rep_number: int
    was_valid: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    duration_frames: int = 0
    feedback: str = ""

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "track_id": self.track_id,
            "exercise_name": self.exercise_name,
            "rep_number": self.rep_number,
            "was_valid": self.was_valid,
            "metrics": dict(self.metrics),
            "duration_frames": self.duration_frames,
            "feedback": self.feedback,
        }
