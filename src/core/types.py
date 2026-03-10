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
