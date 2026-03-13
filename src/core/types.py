from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np

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
    Captura keypoints raw + estado de TODOS los ejercicios activos.
    """
    timestamp: float
    frame_number: int
    track_id: int
    bbox: Tuple[float, float, float, float]
    keypoints_xy: np.ndarray                         # (17, 2)
    keypoints_conf: np.ndarray                       # (17,)
    fps: float                                       # ← nuevo: frames per second
    exercises: Dict[str, ExerciseSnapshot] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialización JSON-safe (todos los valores numpy → Python nativo)."""
        return {
            "timestamp": float(self.timestamp),
            "frame_number": int(self.frame_number),
            "track_id": int(self.track_id),
            "fps": float(self.fps),
            "bbox": [float(v) for v in self.bbox],
            "keypoints_xy": self.keypoints_xy.tolist(),
            "keypoints_conf": self.keypoints_conf.tolist(),
            "exercises": {k: v.to_dict() for k, v in self.exercises.items()},
        }

    def to_csv_row(self) -> list:
        """Fila plana para CSV: metadatos + 51 cols keypoints + ejercicios."""
        row = [
            self.timestamp,
            self.frame_number,
            self.track_id,
            self.fps,
            *self.bbox,
        ]
        # 17 keypoints × 3 (x, y, conf) = 51 columnas
        for i in range(17):
            row.extend([
                float(self.keypoints_xy[i, 0]),
                float(self.keypoints_xy[i, 1]),
                float(self.keypoints_conf[i]),
            ])
        return row

    @staticmethod
    def csv_header() -> list[str]:
        """Nombres de columna para CSV."""
        header = ["timestamp", "frame_number", "track_id", "fps",
                  "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
        for i in range(17):
            header.extend([f"kp{i}_x", f"kp{i}_y", f"kp{i}_conf"])
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

